# quant-bench

Compare un même LLM en **fp32 / fp16 / int8 / 4bit** sur VRAM (pic alloué),
débit (tokens/s) et **énergie consommée** (via
[`energy_measurement`](energy_measurement/README.md), activée par défaut),
avec en option la perplexité (WikiText-2) et IFEval (instruction-following,
via `lm-evaluation-harness`). Si plusieurs GPUs sont visibles, les variantes
tournent en parallèle, une par GPU.

## Installation

```bash
pip install -r requirements.txt
```

`bitsandbytes` a besoin d'un GPU NVIDIA. `lm-eval[ifeval]` (inclus dans
`requirements.txt`) n'est utile que si tu passes `--ifeval`. Testé sur
A100-SXM4-40GB (Ampere).

## Utilisation

```bash
# les 4 variantes : VRAM + tok/s seulement (ppl et IFEval sont désactivés par défaut)
python benchmark.py --model Qwen/Qwen2.5-1.5B

# + perplexité WikiText-2 (test complet, ~330k tokens, quelques min/variante)
python benchmark.py --model Qwen/Qwen2.5-1.5B --ppl

# perplexité tronquée, pour itérer plus vite
python benchmark.py --model Qwen/Qwen2.5-1.5B --ppl --max-tokens 4000

# + IFEval (541 prompts complets, lent — voir "IFEval" plus bas)
python benchmark.py --model Qwen/Qwen2.5-1.5B --ifeval

# IFEval sous-échantillonné, pour un aperçu rapide (score moins fiable)
python benchmark.py --model Qwen/Qwen2.5-1.5B --ifeval --ifeval-limit 40

# sous-ensemble de variantes
python benchmark.py --model meta-llama/Llama-3.2-1B --variants fp16 4bit

# désactiver la mesure d'énergie (itération rapide, ou GPU non libre)
python benchmark.py --model Qwen/Qwen2.5-1.5B --no-energy
```

### Multi-GPU

```bash
./exec.sh                    # transmet ses args à benchmark.py, GPUs 0,1,2,4, une variante par GPU
./exec.sh --ppl --ifeval
```

Voir [Parallélisation](#parallélisation-multi-gpu) plus bas.

### Options principales

| Flag | Défaut | Effet |
|---|---|---|
| `--model` | `Qwen/Qwen2.5-1.5B` | Modèle HF à évaluer |
| `--variants` | les 4 | Sous-ensemble parmi `fp32 fp16 int8 4bit` |
| `--ppl` | désactivé | Calcule la perplexité sur WikiText-2 (test) |
| `--max-tokens` | `0` (tout) | Tronque le texte d'éval ppl à N tokens |
| `--ifeval` | désactivé | Calcule IFEval via lm-evaluation-harness |
| `--ifeval-limit` | tous (541) | Sous-échantillonne IFEval pour aller plus vite |
| `--ifeval-batch-size` | `4` | Batch size pour la génération IFEval |
| `--gen-tokens` | `128` | Longueur de génération pour la mesure de débit |
| `--gpus` | `CUDA_VISIBLE_DEVICES` ou tous | GPUs physiques à utiliser en parallèle |
| `--sequential` | auto | Force l'exécution séquentielle (désactive le multi-GPU) |
| `--out` | `results/<model>_<timestamp>.json` | Fichier de résultats |
| `--no-energy` | désactivé (mesure activée par défaut) | Coupe la mesure d'énergie (`EnergyMeasurement`) |
| `--energy-gpu-index` | premier GPU de `--gpus` | Index physique nvidia-smi à surveiller, en séquentiel seulement |
| `--energy-out` | `results/energy/<model>/<variant>/` | Dossier racine des traces d'énergie |

Sortie console type (avec `--ppl --ifeval`) :

```
variant        ppl   VRAM_GB    tok/s  energy_Wh   avg_W   ifeval   gpu
------------------------------------------------------------------------
fp32        10.213      8.81     34.9      0.412     187     54.3%   0
fp16        10.212      6.28     34.3      0.301     174     53.8%   1
int8        10.278      4.97      8.6      0.256     146     52.9%   2
4bit        11.837      4.40     24.6      0.198     139     47.5%   4
```

## Méthode

- **VRAM** — `torch.cuda.max_memory_allocated()`, mesuré après toute
  l'évaluation de la variante (débit + ppl + IFEval si activés).
- **Énergie** (activée par défaut, `--no-energy` pour désactiver) — chaque
  variante charge son modèle et fait une petite génération de chauffe hors
  mesure, puis le débit (et la perplexité si `--ppl`) tournent sous
  [`EnergyMeasurement`](energy_measurement/README.md) : joules intégrés sur
  la trace `nvidia-smi` réelle, watts moyens, utilisation GPU, VRAM. IFEval
  reste **hors mesure d'énergie** : `lm-evaluation-harness` fait ses propres
  I/O (téléchargement/chargement de données) pendant l'évaluation, ce qui
  fausserait la trace de puissance (voir le protocole détaillé dans
  `energy_measurement/README.md`). Chaque variante écrit sa trace dans
  `results/energy/<model>/<variant>/<timestamp>/` (`power_trace.csv`,
  `energy_timeseries.csv`, `summary.json`).
- **Débit** — génération greedy de 128 tokens, `synchronize()` autour du chrono.
- **Perplexité** (`--ppl`) — fenêtre glissante sur WikiText-2 test (stride 512,
  contexte 2048), seuls les nouveaux tokens de chaque fenêtre sont scorés.
  Protocole HF standard, comparable entre variantes puisque le checkpoint est
  identique.
- **IFEval** (`--ifeval`) — instruction-following, via `lm_eval.simple_evaluate`
  sur le modèle déjà chargé (pas de rechargement). Métrique reportée :
  `prompt_level_strict_acc`. Le détail complet (les 4 sous-métriques IFEval) est
  conservé dans le JSON de sortie sous `results[].ifeval`.

## Parallélisation multi-GPU

Si plusieurs GPUs sont visibles (`CUDA_VISIBLE_DEVICES`/`--gpus`) et plusieurs
variantes demandées, chaque variante tourne dans son propre sous-process
Python, un GPU dédié chacune, en parallèle (sinon, exécution séquentielle
classique dans le process principal — utile pour debug avec `--sequential`).

- Les logs de chaque sous-process (barres `tqdm` comprises) vont dans
  `logs/<variant>.log`, pas mélangés au terminal.
- Les résultats sont réordonnés selon `--variants` avant affichage.
- Une variante en échec (OOM, etc.) apparaît en `ERREUR` dans le tableau sans
  bloquer les autres.
- `logs/` et `results/` sont dans `.gitignore` (générés, pas versionnés).

⚠️ En parallèle, `tok/s` est moins fiable pour comparer les variantes entre
elles : les 4 sous-process se partagent des ressources hôte (CPU, PCIe), ce
qui compresse les écarts de débit réels. Pour un chiffre de débit rigoureux,
relance avec `--sequential`.

⚠️ `EnergyMeasurement` exige un GPU **libre** (aucun autre process dessus) au
moment d'entrer dans le bloc mesuré — voir
[energy_measurement/README.md](energy_measurement/README.md). En parallèle
chaque sous-process cible automatiquement son propre GPU physique
(`CUDA_VISIBLE_DEVICES` par variante), donc c'est transparent. En séquentiel
avec plusieurs GPUs visibles, la mesure d'énergie ne surveille que le premier
(`--energy-gpu-index` pour en choisir un autre) puisque tout tourne dans le
même process sur un seul device à la fois. Un run court (peu de tokens, pas
de `--ppl`) donne aussi une fenêtre de mesure très brève : la puissance
moyenne est alors plus bruitée (voir "Bloc d'au moins 60 secondes recommandé"
dans `energy_measurement/README.md`).

## À savoir sur l'A100

- **int8 (bnb)** est souvent *plus lent* que fp16 en génération malgré moins de
  VRAM : la méthode LLM.int8() isole les outliers et calcule une partie en
  fp16 (overhead). C'est aussi la variante la plus lente sur IFEval (beaucoup
  de génération) — attends-toi à ce qu'elle domine le temps total d'un run
  `--ifeval` complet.
- **fp8** n'est pas inclus : pas de tensor cores FP8 sur Ampere (Hopper/H100
  requis). En weight-only il tournerait mais sans gain de vitesse.
- **fp32 vs fp16** : perplexité quasi identique, parfois strictement égale sur
  un petit modèle. fp32 sert surtout de baseline « pleine précision » ; le
  contraste net se voit surtout en 4bit.

## Machine(s) utilisée(s)

- GPUs 0, 1, 2, 4 disponibles ; **GPU 3 interdit** (réservé/hors limites) —
  filtré automatiquement (`FORBIDDEN_GPUS` dans `benchmark.py`) même si un
  lancement direct sans `exec.sh` l'incluait par erreur.
- `CUDA_DEVICE_ORDER=PCI_BUS_ID` est fixé (dans `exec.sh` et en fallback dans
  `benchmark.py`) pour que la numérotation CUDA corresponde à celle de
  `nvidia-smi` — sans ça, les indices de `CUDA_VISIBLE_DEVICES` peuvent
  pointer vers un GPU physique différent de celui attendu.

## Variables d'ajustement

- `--max-tokens` : réduit fortement le temps d'éval ppl (utile pour itérer).
- `--ifeval-limit` : idem pour IFEval (541 prompts par défaut, sous-échantillonne
  pour un score approximatif plus rapide).
- `--gen-tokens` : longueur de génération pour la mesure de débit.
- Modèles conseillés (petits, dispos partout) : `Qwen/Qwen2.5-0.5B`,
  `Qwen/Qwen2.5-1.5B`, `meta-llama/Llama-3.2-1B`, `google/gemma-2-2b`.
