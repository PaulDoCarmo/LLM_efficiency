# quant-bench

Compare un même LLM en **fp32 / fp16 / int8 / 4bit** sur trois axes :
perplexité (qualité), VRAM (pic alloué) et débit (tokens/s).

## Installation

```bash
pip install -r requirements.txt
```

`bitsandbytes` a besoin d'un GPU NVIDIA. Testé sur A100-SXM4-40GB (Ampere).

## Utilisation

```bash
# les 4 variantes, WikiText-2 complet (~330k tokens, quelques min/variante)
python benchmark.py --model Qwen/Qwen2.5-1.5B

# aperçu rapide : tronque l'éval à 4000 tokens
python benchmark.py --model Qwen/Qwen2.5-1.5B --max-tokens 4000

# sous-ensemble de variantes
python benchmark.py --model meta-llama/Llama-3.2-1B --variants fp16 4bit
```

Sortie type :

```
variant        ppl   VRAM_GB    tok/s
--------------------------------------
fp32        12.345      6.01     22.4
fp16        12.348      3.01     41.8
int8        12.361      1.83     28.6
4bit        12.720      1.24     47.2
```

## Méthode

- **Perplexité** — fenêtre glissante sur WikiText-2 test (stride 512, contexte 2048),
  seuls les nouveaux tokens de chaque fenêtre sont scorés. C'est le protocole HF
  standard, comparable entre variantes puisque le checkpoint est identique.
- **VRAM** — `torch.cuda.max_memory_allocated()` après chargement + inférence.
- **Débit** — génération greedy de 128 tokens, `synchronize()` autour du chrono.

## À savoir sur l'A100

- **int8 (bnb)** est souvent *plus lent* que fp16 malgré moins de VRAM : la méthode
  LLM.int8() isole les outliers et calcule une partie en fp16 (overhead). Pour de
  l'int8 rapide sur A100, voir `torchao`.
- **fp8** n'est pas inclus : pas de tensor cores FP8 sur Ampere (Hopper/H100 requis).
  En weight-only il tournerait mais sans gain de vitesse.
- **fp32 vs fp16** : perplexité quasi identique. fp32 sert surtout de baseline
  « pleine précision » ; le contraste net se voit surtout en 4bit.

## Variables d'ajustement

- `--max-tokens` : réduit fortement le temps d'éval (utile pour itérer).
- `--gen-tokens` : longueur de génération pour la mesure de débit.
- Modèles conseillés (petits, dispos partout) : `Qwen/Qwen2.5-0.5B`,
  `Qwen/Qwen2.5-1.5B`, `meta-llama/Llama-3.2-1B`, `google/gemma-2-2b`.