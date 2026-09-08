# Mesure d'énergie GPU

## Installation

```bash
pip install -r requirements.txt
```

## Usage minimal

```python
from energy_measurement import EnergyMeasurement

with EnergyMeasurement(gpu_index=0, metadata={"model": "smollm2-360m", "quant": "bf16"}) as m:
    # Uniquement la boucle d'inférence continue à mesurer ici.
    # Pas de print, pas d'I/O, pas de chargement de données dans ce bloc :
    # ça fausse la mesure de puissance (voir CLAUDE.md).
    run_my_benchmark()

print(m.energy_j)               # énergie consommée, en joules
print(m.energy_j / 3600)        # ... en Wh
print(m.mean_power_w)           # puissance moyenne, en watts
print(m.mean_utilization_pct)   # utilisation GPU moyenne, en %
print(m.mean_vram_mib)          # VRAM moyenne utilisée, en MiB
print(m.peak_vram_mib)          # VRAM max utilisée, en MiB
```

Tout est déjà calculé à la sortie du bloc `with` — rien à relancer après.

## Ce qu'il faut savoir avant d'appeler ça

- **Un GPU par process.** Passer l'index du GPU à utiliser (`gpu_index=0`,
  `1`, `2` ou `4` sur la DGX Station de l'équipe). Le GPU 3 est un GPU
  d'affichage, il est bloqué automatiquement.
- **Le GPU doit être libre.** `EnergyMeasurement(...)` vérifie qu'aucun
  autre processus ne tourne déjà dessus et s'arrête avec un message clair
  sinon (la puissance NVML est mesurée par carte, pas par processus — un
  job concurrent fausserait la trace).
- **Faire la chauffe *avant* d'entrer dans le `with`.** Le premier appel au
  modèle est toujours plus lent (compilation de kernels, allocations) ;
  l'inclure dans la mesure biaiserait le résultat.
- **Rien d'autre que du calcul GPU dans le `with`.** Pas de `print`, pas de
  chargement de données, pas d'écriture de fichier. `power.draw` de
  `nvidia-smi` est une moyenne glissante sur 1 seconde : la moindre pause
  crée un creux qui sous-estime la puissance réelle.
- **Bloc d'au moins 60 secondes recommandé.** En dessous, la mesure est
  trop bruitée pour être fiable.

## Fichiers produits

Chaque appel crée un dossier `results/<timestamp>/` avec :

| Fichier                 | Contenu                                                        |
|--------------------------|-----------------------------------------------------------------|
| `power_trace.csv`        | Trace brute nvidia-smi (100 ms) : puissance, horloge, temp, VRAM |
| `energy_timeseries.csv`  | Une ligne par timestamp : puissance, utilisation, énergie cumulée |
| `summary.json`           | Résultats agrégés + métadonnées passées à `metadata=`          |
| `pip_freeze.txt`         | Versions des packages installés au moment du run                |

## Tester que ça marche

```bash
python measure_dummy_inference.py --duration 60 --gpu-index 0
```

Ou via le wrapper qui source le venv du projet automatiquement :

```bash
./run_test.sh [gpu_index] [duration_s]   # défaut : GPU 0, 60s
./run_test.sh 1 30                       # GPU 1, 30s
```

Lance un faux modèle PyTorch sur le GPU choisi et affiche les résultats à la
fin. Sert à vérifier que `energy_measurement.py` fonctionne avant de le
brancher sur un vrai benchmark.
