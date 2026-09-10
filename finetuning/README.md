# Finetuning QLoRA pour le suivi d'instructions

Finetune `Qwen/Qwen2.5-1.5B` **quantifié en 4bit** (QLoRA) sur
[`allenai/tulu-3-sft-personas-instruction-following`](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following),
puis évalue le gain sur **IFEval**.

## Installation

En plus des dépendances du projet (`requirements.txt` à la racine) :

```bash
pip install -r requirements.txt   # inclut peft
```

## Utilisation

```bash
# 0. référence : IFEval sur le modèle de base, AVANT finetuning
python finetuning/eval_ifeval.py --limit 100

# 1. essai rapide pour valider que la boucle tourne (~quelques minutes)
python finetuning/finetune_lora.py --max-samples 2000 --epochs 1

# 2. entraînement complet
python finetuning/finetune_lora.py

# 3. IFEval sur le modèle finetuné
python finetuning/eval_ifeval.py --adapter finetuning/out/qwen2.5-1.5b-qlora-ifeval
```

Pour choisir le GPU (rappel : **GPU 3 interdit** sur cette machine) :

```bash
CUDA_VISIBLE_DEVICES=0 python finetuning/finetune_lora.py
```

## Recette

| Élément | Choix | Pourquoi |
|---|---|---|
| Quantification base | NF4 + double quant, calcul bf16 | Recette QLoRA standard. NF4 ≠ fp4 (le défaut de bitsandbytes, utilisé par la variante `4bit` de `benchmark.py`) : niveaux placés sur les quantiles d'une gaussienne, mieux adaptés à des poids normalement distribués. |
| LoRA | r=16, alpha=32, dropout=0.05 | Réglage courant pour un 1.5B. Cible les projections attention (`q,k,v,o_proj`) **et** MLP (`gate,up,down_proj`). |
| Perte | Réponse assistant uniquement | Le prompt est masqué à `-100`. C'est ce qu'on veut en instruction tuning : le modèle apprend à répondre, pas à régurgiter la consigne. |
| Optimiseur | `paged_adamw_8bit` | Optimiseur paginé bitsandbytes, complète la recette QLoRA. |
| Précision | bf16 | Précision native d'entraînement de Qwen, et native sur A100. |
| `use_reentrant=False` | gradient checkpointing | Requis pour que le checkpointing coopère avec PEFT. |

Seul l'**adaptateur** est sauvegardé (quelques dizaines de Mo) ; la base 4bit est
rechargée depuis le Hub à l'évaluation.

## Point d'attention : cohérence de formatage

L'entraînement formate les exemples avec le **chat template** de Qwen
(`<|im_start|>user ... <|im_start|>assistant ...`). Or les prompts IFEval de
lm-evaluation-harness sont des instructions **brutes**.

Si on entraîne avec template et qu'on évalue sans, le modèle est hors
distribution et le score s'effondre. `eval_ifeval.py` applique donc le chat
template **automatiquement dès qu'un `--adapter` est passé**, et pas pour le
modèle de base.

Conséquence : avec les réglages par défaut, la base et le finetuné ne sont pas
évalués dans des conditions strictement identiques, et une partie du gain
mesuré viendrait alors du **formatage**, pas de l'apprentissage. Pour isoler
les deux effets, évalue la base dans les deux formats :

```bash
python finetuning/eval_ifeval.py --limit 100                    # base, prompts bruts
python finetuning/eval_ifeval.py --limit 100 --chat-template    # base, avec template
python finetuning/eval_ifeval.py --limit 100 --adapter finetuning/out/...   # finetuné
```

Le gain réellement attribuable au finetuning, c'est l'écart entre la 3ᵉ ligne
et la 2ᵉ (même formatage des deux côtés).

## Modèle fusionné en 4bit → IFEval de `benchmark.py`

`eval_merged_4bit.py` fusionne l'adaptateur dans la base, requantifie en 4bit,
et appelle **le `run_ifeval` de `benchmark.py`** (import direct, pas de copie).

**Dans quoi fusionner ?** C'est le point délicat. Pendant l'entraînement le
forward est `y = dequant(W_nf4)·x + ΔW·x` : l'adaptateur a appris contre la base
**quantifiée**, et une partie de ce que `ΔW` encode est la compensation de
l'erreur de quantification. D'où deux stratégies :

| `--merge-into` | Calcul | |
|---|---|---|
| `quantized` (défaut) | `quantize(dequant(W_nf4) + ΔW)` | **Fidèle aux conditions d'entraînement.** PEFT déquantifie, additionne, requantifie couche par couche. Pas de détour par le disque. |
| `bf16` | `quantize(W + ΔW)` | Fusion dans les poids d'origine. Plus fidèle à `W`, mais pas à ce que l'adaptateur a appris. Passe par le disque (~3 Go temporaires). |

Le défaut `quantized` est le bon choix ; `bf16` est là pour pouvoir comparer les
deux et mesurer l'écart.

```bash
# tester le pipeline AVANT la fin de l'entraînement
python finetuning/eval_merged_4bit.py --dummy --limit 5

# une fois l'entraînement fini
python finetuning/eval_merged_4bit.py --adapter finetuning/out/qwen2.5-1.5b-qlora-ifeval --limit 100
```

En `--dummy`, l'adaptateur LoRA est créé non entraîné : sa matrice `B` est
initialisée à zéro, donc `B@A = 0` et la fusion est un **no-op mathématique**.
Le score doit donc reproduire celui de la base quantifiée — c'est ce qui rend
ce mode utile comme vérification, et pas seulement comme test anti-crash.

`--quant-type` : `nf4` (défaut, celle de l'entraînement) ou `fp4` (celle de la
variante `4bit` de `benchmark.py`, pour une comparaison stricte avec cette
ligne du tableau).

⚠️ `run_ifeval` de `benchmark.py` n'applique **pas** le chat template, alors
que le finetuning en utilise un. Le score sera donc pessimiste pour le modèle
finetuné. Utilise `eval_ifeval.py --adapter ...` (qui applique le template)
pour mesurer le vrai gain, et `eval_merged_4bit.py` pour comparer au tableau
de `benchmark.py`.

## Lien avec `benchmark.py`

`benchmark.py` (racine) compare des **quantifications** du même checkpoint
(fp16/bf16/int8/4bit) sur VRAM, débit, énergie, IFEval. Ici on fait l'inverse :
on fixe la quantification (4bit) et on fait **bouger les poids** via LoRA.

Les deux scripts partagent la même métrique IFEval
(`prompt_level_strict_acc` via lm-evaluation-harness), donc les scores sont
comparables — à condition d'utiliser le même réglage `apply_chat_template`.

## Ce qui n'a pas été testé

Ces scripts n'ont **pas** été exécutés (pas de GPU sur la machine de
développement) : ils compilent, mais la première exécution sur l'A100 peut
demander des ajustements (versions `peft`/`trl`, empreinte VRAM, schéma exact
du dataset). Commence par `--max-samples 2000 --epochs 1`.
