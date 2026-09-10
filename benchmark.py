#!/usr/bin/env python
"""Compare un LLM en fp16 / bf16 / int8 / 4bit / int4 : VRAM, débit, et en option
perplexité / IFEval.

"4bit" = bitsandbytes fp4 (float 4 bits, niveaux non uniformes).
"int4"  = torchao int4 weight-only (vrai entier 4 bits uniforme, par groupes).

VRAM = pic alloué. tok/s = génération greedy de 128 tokens. Perplexité (--ppl,
désactivée par défaut) calculée sur WikiText-2 (test). Pensé pour un GPU
Ampere+ (testé sur A100-40GB).

Si plusieurs GPUs sont visibles (CUDA_VISIBLE_DEVICES ou --gpus) et plusieurs
variantes demandées, chaque variante tourne dans son propre sous-process,
un GPU dédié chacune, en parallèle. Sinon, exécution séquentielle classique
dans ce process.

Usage:
    python benchmark.py --model Qwen/Qwen2.5-1.5B
    python benchmark.py --model meta-llama/Llama-3.2-1B --variants fp16 4bit --max-tokens 4000
    CUDA_VISIBLE_DEVICES=0,1,2,4 python benchmark.py   # 4 variantes en parallèle, une par GPU
    python benchmark.py --ifeval --ifeval-limit 40     # + score IFEval (sous-échantillonné)
    python benchmark.py --ppl                          # + perplexité WikiText-2
    python benchmark.py --variants int8 int4 --ifeval  # compare seulement int8 et int4
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from queue import Queue

# Aligne la numérotation CUDA sur celle de nvidia-smi (sinon l'ordre "carte la
# plus rapide d'abord" de CUDA peut ne pas correspondre aux indices attendus,
# et un GPU qu'on croit exclure via CUDA_VISIBLE_DEVICES n'est pas le bon).
# Doit être fait avant le premier appel torch.cuda.*.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch

warnings.filterwarnings("ignore", message="MatMul8bitLt: inputs will be cast")
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).parent / "energy_measurement"))
from energy_measurement import EnergyMeasurement  # noqa: E402

ALL_VARIANTS = ["fp16", "bf16", "int8", "4bit", "int4"]
FORBIDDEN_GPUS = {"3"}  # GPU 3 hors limites sur cette machine, ne jamais l'utiliser.
WARMUP_GEN_TOKENS = 8  # chauffe hors mesure, avant d'entrer dans EnergyMeasurement.
IFEVAL_WARMUP_LIMIT = 1  # idem, pour chauffer le cache dataset IFEval + compiler les kernels.


def build_configs(selected):
    """Config `from_pretrained` par variante. Construit uniquement celles
    demandées : ça évite d'importer torchao (variante int4) si on ne s'en
    sert pas."""

    def _make(name):
        if name == "fp16":
            return dict(dtype=torch.float16)
        if name == "bf16":
            return dict(dtype=torch.bfloat16)
        if name == "int8":
            return dict(
                dtype=torch.float16,
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            )
        if name == "4bit":  # bitsandbytes fp4 : float 4 bits, niveaux non uniformes
            return dict(
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
                )
            )
        if name == "int4":
            # torchao int4 entier uniforme, par groupes. Appliqué APRÈS
            # from_pretrained (voir _apply_torchao_int4) : le pont
            # transformers TorchAoConfig utilise les anciens noms d'API
            # (int4_weight_only/autoquant), supprimés dans torchao >= 0.14.
            return dict(dtype=torch.bfloat16)
        raise KeyError(name)

    return {k: _make(k) for k in selected}


def _apply_torchao_int4(model, group_size=128):
    """Quantifie le modèle en int4 weight-only (entier uniforme, par groupes)
    via torchao, en place. API moderne (Int4WeightOnlyConfig + quantize_) pour
    rester compatible torchao >= 0.14 ; version=1 (kernel tinygemm) demandé
    explicitement car version 2 vise des GPU Hopper."""
    from torchao.quantization import Int4WeightOnlyConfig, quantize_

    try:
        cfg = Int4WeightOnlyConfig(group_size=group_size, version=1)
    except TypeError:  # torchao trop ancien pour le paramètre version
        cfg = Int4WeightOnlyConfig(group_size=group_size)
    quantize_(model, cfg)


@torch.no_grad()
def perplexity(model, enc, stride=512, max_len=2048):
    """Perplexité par fenêtre glissante (approche standard HF)."""
    nlls = []
    for i in range(0, enc.input_ids.size(1), stride):
        ids = enc.input_ids[:, i : i + max_len].to(model.device)
        tgt = ids.clone()
        tgt[:, :-stride] = -100  # ne scorer que les nouveaux tokens
        nlls.append(model(ids, labels=tgt).loss)
    return torch.exp(torch.stack(nlls).mean()).item()


def throughput(model, tok, n=128):
    ids = tok("The capital of France is", return_tensors="pt").input_ids.to(model.device)
    torch.cuda.synchronize()
    t0 = time.time()
    out = model.generate(ids, max_new_tokens=n, do_sample=False)
    torch.cuda.synchronize()
    return (out.size(1) - ids.size(1)) / (time.time() - t0)


def _ifeval_batch_size(v):
    """--ifeval-batch-size accepte un entier ou 'auto' (lm-eval cherche alors
    le plus grand batch qui tient dans la VRAM libre)."""
    return v if v in ("auto",) or str(v).startswith("auto:") else int(v)


def _extract_metric(metrics, name):
    """lm-eval préfixe les clés avec le nom du filtre (ex: 'prompt_level_strict_acc,none')."""
    for k, v in metrics.items():
        if k.split(",")[0] == name:
            return v
    return None


def run_ifeval(model, tok, limit=None, batch_size="auto"):
    """Évalue IFEval (instruction-following) sur le modèle déjà chargé, via
    lm-evaluation-harness. Import différé : lm_eval reste optionnel tant
    qu'on ne passe pas --ifeval."""
    import logging

    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    logging.getLogger("lm-eval").setLevel(logging.WARNING)
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch_size)
    out = simple_evaluate(model=lm, tasks=["ifeval"], limit=limit, bootstrap_iters=0)
    return out["results"]["ifeval"]


def energy_output_dir(model_name, variant, base="results/energy"):
    return Path(base) / model_name.replace("/", "_") / variant


def run_variant(
    model_name,
    variant,
    max_tokens,
    gen_tokens,
    compute_ppl=False,
    ifeval=False,
    ifeval_limit=None,
    ifeval_batch_size="auto",
    measure_energy=True,
    energy_gpu_index=0,
    energy_dir=None,
):
    """Charge tokenizer+modèle et évalue UNE variante. Suppose que
    CUDA_VISIBLE_DEVICES est déjà positionné correctement par l'appelant.

    Si measure_energy est activé, le débit, la perplexité (si demandée) et
    IFEval (si demandé) tournent tous sous EnergyMeasurement, sur
    `energy_gpu_index` (index PHYSIQUE nvidia-smi du GPU réellement utilisé
    par ce process — voir le README de energy_measurement/). Comme
    lm-evaluation-harness télécharge/charge son dataset et compile des
    kernels de génération au premier appel, un passage IFEval "à vide" (un
    seul exemple) est fait hors mesure juste avant, uniquement pour chauffer
    ce cache — le run IFEval réel (celui compté dans les résultats) a lieu
    dans le bloc mesuré."""
    tok = AutoTokenizer.from_pretrained(model_name)

    cfg = build_configs([variant])[variant]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", **cfg)
    if variant == "int4":
        _apply_torchao_int4(model)
        # Le modèle est chargé en bf16 puis packé en int4 : on ré-arme la mesure
        # de pic après packing pour ne pas compter le pic transitoire du bf16
        # (les variantes bnb, elles, quantifient déjà pendant from_pretrained).
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model.eval()

    # Toute I/O (chargement de données) doit être terminée avant d'entrer
    # dans le bloc mesuré par EnergyMeasurement.
    ppl_enc = None
    if compute_ppl:
        text = "\n\n".join(
            load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
        )
        ppl_enc = tok(text, return_tensors="pt")
        if max_tokens:
            ppl_enc.input_ids = ppl_enc.input_ids[:, :max_tokens]

    # Chauffe hors mesure : le premier appel au modèle compile des kernels et
    # alloue de la mémoire, ce qui fausserait aussi bien tok/s que la trace
    # de puissance si on le laissait dans le bloc mesuré.
    throughput(model, tok, WARMUP_GEN_TOKENS)
    if ifeval:
        # Idem pour IFEval : premier appel = téléchargement/chargement du
        # dataset + compilation. Un seul exemple suffit à chauffer le cache
        # (le dataset entier est mis en cache local dès ce premier appel).
        run_ifeval(model, tok, limit=IFEVAL_WARMUP_LIMIT, batch_size=ifeval_batch_size)
    torch.cuda.synchronize()

    def _run_ifeval_and_record(res):
        ifeval_metrics = run_ifeval(model, tok, limit=ifeval_limit, batch_size=ifeval_batch_size)
        res["ifeval"] = ifeval_metrics
        res["ifeval_score"] = _extract_metric(ifeval_metrics, "prompt_level_strict_acc")

    result = {"variant": variant}

    if measure_energy:
        run_dir = energy_dir or energy_output_dir(model_name, variant)
        with EnergyMeasurement(
            gpu_index=energy_gpu_index,
            output_dir=run_dir,
            metadata={"model": model_name, "variant": variant, "gen_tokens": gen_tokens},
        ) as em:
            result["tok_s"] = throughput(model, tok, gen_tokens)
            if compute_ppl:
                result["ppl"] = perplexity(model, ppl_enc)
            if ifeval:
                _run_ifeval_and_record(result)
        result["energy_j"] = em.energy_j
        result["energy_wh"] = em.energy_j / 3600.0
        result["mean_power_w"] = em.mean_power_w
        result["mean_utilization_pct"] = em.mean_utilization_pct
        result["peak_vram_mib"] = em.peak_vram_mib
        result["mean_vram_mib"] = em.mean_vram_mib
        result["energy_run_dir"] = str(em.run_dir)
    else:
        result["tok_s"] = throughput(model, tok, gen_tokens)
        if compute_ppl:
            result["ppl"] = perplexity(model, ppl_enc)
        if ifeval:
            _run_ifeval_and_record(result)

    # Mesuré en dernier : capture le pic mémoire de toute l'évaluation (ppl + tok/s + ifeval).
    result["vram_gb"] = torch.cuda.max_memory_allocated() / 1e9

    del model
    torch.cuda.empty_cache()
    return result


def gpu_list_from_env():
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if raw.strip():
        return [g.strip() for g in raw.split(",") if g.strip()]
    if torch.cuda.is_available():
        return [str(i) for i in range(torch.cuda.device_count())]
    return []


def run_variant_subprocess(
    model_name,
    variant,
    max_tokens,
    gen_tokens,
    gpu_id,
    result_path,
    log_path,
    compute_ppl=False,
    ifeval=False,
    ifeval_limit=None,
    ifeval_batch_size="auto",
    measure_energy=True,
    energy_out=None,
):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--model", model_name,
        "--max-tokens", str(max_tokens),
        "--gen-tokens", str(gen_tokens),
        "--worker-variant", variant,
        "--result-file", str(result_path),
        # CUDA_VISIBLE_DEVICES ne réduit qu'aux yeux de torch : nvidia-smi
        # (utilisé par EnergyMeasurement) voit toujours l'index physique.
        "--energy-gpu-index", gpu_id,
    ]
    if compute_ppl:
        cmd.append("--ppl")
    if ifeval:
        cmd.append("--ifeval")
        if ifeval_limit is not None:
            cmd += ["--ifeval-limit", str(ifeval_limit)]
        cmd += ["--ifeval-batch-size", str(ifeval_batch_size)]
    if not measure_energy:
        cmd.append("--no-energy")
    if energy_out:
        cmd += ["--energy-out", str(energy_out)]

    print(f"[{variant}] démarré sur GPU {gpu_id}", flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0 or not result_path.exists():
        result = {"variant": variant, "error": f"échec (code {proc.returncode}), voir {log_path}"}
        print(f"[{variant}] terminé (ÉCHEC, voir {log_path})", flush=True)
        return result
    with open(result_path, encoding="utf-8") as f:
        result = json.load(f)
    result["gpu"] = gpu_id
    result_path.unlink(missing_ok=True)
    extra = ""
    if result.get("ppl") is not None:
        extra += f" ppl={result['ppl']:.3f}"
    if result.get("energy_wh") is not None:
        extra += f" energy={result['energy_wh']:.3f}Wh ({result['mean_power_w']:.0f}W moy.)"
    if result.get("ifeval_score") is not None:
        extra += f" ifeval={result['ifeval_score'] * 100:.1f}%"
    print(
        f"[{variant}] terminé sur GPU {gpu_id} : "
        f"vram={result['vram_gb']:.2f}GB tok/s={result['tok_s']:.1f}{extra}",
        flush=True,
    )
    return result


def run_parallel(
    model_name,
    variants,
    max_tokens,
    gen_tokens,
    gpus,
    tmp_dir,
    log_dir,
    compute_ppl=False,
    ifeval=False,
    ifeval_limit=None,
    ifeval_batch_size="auto",
    measure_energy=True,
    energy_out=None,
):
    gpu_queue: Queue = Queue()
    for g in gpus:
        gpu_queue.put(g)

    def worker(variant):
        gpu_id = gpu_queue.get()
        try:
            result_path = tmp_dir / f"{variant}.json"
            log_path = log_dir / f"{variant}.log"
            return run_variant_subprocess(
                model_name,
                variant,
                max_tokens,
                gen_tokens,
                gpu_id,
                result_path,
                log_path,
                compute_ppl=compute_ppl,
                ifeval=ifeval,
                ifeval_limit=ifeval_limit,
                ifeval_batch_size=ifeval_batch_size,
                measure_energy=measure_energy,
                energy_out=energy_out,
            )
        finally:
            gpu_queue.put(gpu_id)

    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        return list(ex.map(worker, variants))


def print_table(results, ifeval=False):
    header = f"\n{'variant':8} {'ppl':>9} {'VRAM_GB':>9} {'tok/s':>8} {'energy_Wh':>10} {'avg_W':>7}"
    if ifeval:
        header += f" {'ifeval':>8}"
    header += "   gpu"
    print(header)
    print("-" * (len(header) + 4))
    lines = []
    for r in results:
        if "error" in r:
            line = f"{r['variant']:8} {'ERREUR':>9}   {r['error']}"
        else:
            ppl = r.get("ppl")
            ppl_str = f"{ppl:9.3f}" if ppl is not None else f"{'n/a':>9}"
            energy_wh = r.get("energy_wh")
            energy_str = f"{energy_wh:10.3f}" if energy_wh is not None else f"{'n/a':>10}"
            avg_w = r.get("mean_power_w")
            avg_w_str = f"{avg_w:7.0f}" if avg_w is not None else f"{'n/a':>7}"
            line = (
                f"{r['variant']:8} {ppl_str} {r['vram_gb']:9.2f} {r['tok_s']:8.1f} "
                f"{energy_str} {avg_w_str}"
            )
            if ifeval:
                score = r.get("ifeval_score")
                line += f" {score * 100:7.1f}%" if score is not None else f" {'n/a':>8}"
            line += f"   {r.get('gpu', '-')}"
        print(line)
        lines.append(line)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument(
        "--variants",
        nargs="+",
        default=ALL_VARIANTS,
        choices=ALL_VARIANTS,
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Tronque le texte d'éval à N tokens (0 = tout WikiText-2, ~330k, lent).",
    )
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument(
        "--gpus",
        help="GPUs physiques pour la parallélisation, ex: '0,1,2,4'. "
        "Défaut: CUDA_VISIBLE_DEVICES, sinon tous les GPUs visibles.",
    )
    ap.add_argument(
        "--sequential",
        action="store_true",
        help="Force l'exécution séquentielle dans ce process (désactive la parallélisation).",
    )
    ap.add_argument(
        "--out",
        help="Fichier de résultats JSON. Défaut: results/<model>_<timestamp>.json",
    )
    ap.add_argument(
        "--ppl",
        dest="compute_ppl",
        action="store_true",
        help="Calcule la perplexité sur WikiText-2 (désactivé par défaut, mis de côté pour l'instant).",
    )
    ap.add_argument(
        "--ifeval",
        action="store_true",
        help="Évalue aussi IFEval (instruction-following) via lm-evaluation-harness, "
        "en plus de ppl/VRAM/tok-s. Nécessite lm-eval[ifeval] (voir requirements.txt).",
    )
    ap.add_argument(
        "--ifeval-limit",
        type=int,
        default=None,
        help="Limite le nombre d'exemples IFEval (défaut: tous, 541 prompts, lent).",
    )
    ap.add_argument(
        "--ifeval-batch-size",
        type=_ifeval_batch_size,
        default="auto",
        help="Batch size IFEval : entier, ou 'auto' (défaut) — lm-eval cherche le "
        "plus grand batch qui tient dans la VRAM libre, donc plus gros pour "
        "int4/4bit (modèle plus petit) que pour int8.",
    )
    ap.add_argument(
        "--no-energy",
        dest="measure_energy",
        action="store_false",
        help="Désactive la mesure d'énergie (activée par défaut). Utile pour itérer "
        "vite ou si le GPU n'est pas libre pour EnergyMeasurement (voir "
        "energy_measurement/README.md).",
    )
    ap.add_argument(
        "--energy-gpu-index",
        type=int,
        default=None,
        help="Index PHYSIQUE nvidia-smi du GPU à surveiller pour la mesure d'énergie "
        "en mode séquentiel. Défaut: premier GPU de --gpus/CUDA_VISIBLE_DEVICES. "
        "En mode parallèle chaque sous-process déduit automatiquement le sien.",
    )
    ap.add_argument(
        "--energy-out",
        help="Dossier racine des résultats d'énergie. Défaut: results/energy/<model>/<variant>/.",
    )
    # Flags internes utilisés par les sous-process workers, cachés du --help.
    ap.add_argument("--worker-variant", dest="worker_variant", help=argparse.SUPPRESS)
    ap.add_argument("--result-file", dest="result_file", help=argparse.SUPPRESS)
    ap.set_defaults(measure_energy=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA requis (bitsandbytes ne quantifie que sur GPU).")

    # Mode worker : évalue une seule variante, écrit le résultat en JSON, puis quitte.
    if args.worker_variant:
        result = run_variant(
            args.model,
            args.worker_variant,
            args.max_tokens,
            args.gen_tokens,
            compute_ppl=args.compute_ppl,
            ifeval=args.ifeval,
            ifeval_limit=args.ifeval_limit,
            ifeval_batch_size=args.ifeval_batch_size,
            measure_energy=args.measure_energy,
            energy_gpu_index=args.energy_gpu_index if args.energy_gpu_index is not None else 0,
            energy_dir=Path(args.energy_out) / args.worker_variant if args.energy_out else None,
        )
        with open(args.result_file, "w", encoding="utf-8") as f:
            json.dump(result, f)
        return

    gpus = args.gpus.split(",") if args.gpus else gpu_list_from_env()
    excluded = [g for g in gpus if g in FORBIDDEN_GPUS]
    if excluded:
        print(f"ATTENTION: GPU(s) {excluded} exclu(s) de la liste (interdits sur cette machine).")
        gpus = [g for g in gpus if g not in FORBIDDEN_GPUS]
    parallel = not args.sequential and len(gpus) > 1 and len(args.variants) > 1

    print(f"\nmodel = {args.model}   variants = {args.variants}")
    t0 = time.time()

    if parallel:
        print(f"GPUs = {gpus}   (exécution parallèle, un sous-process par variante)")
        tmp_dir = Path(tempfile.mkdtemp(prefix="bench_"))
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        try:
            results = run_parallel(
                args.model,
                args.variants,
                args.max_tokens,
                args.gen_tokens,
                gpus,
                tmp_dir,
                log_dir,
                compute_ppl=args.compute_ppl,
                ifeval=args.ifeval,
                ifeval_limit=args.ifeval_limit,
                ifeval_batch_size=args.ifeval_batch_size,
                measure_energy=args.measure_energy,
                energy_out=args.energy_out,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        print(f"device = {torch.cuda.get_device_name()}   (exécution séquentielle)")
        # En séquentiel, un seul GPU physique sert réellement torch : le premier
        # de --gpus/CUDA_VISIBLE_DEVICES (sinon --energy-gpu-index explicite).
        seq_energy_gpu_index = (
            args.energy_gpu_index if args.energy_gpu_index is not None
            else int(gpus[0]) if gpus else 0
        )
        if args.measure_energy and len(gpus) > 1:
            print(
                f"ATTENTION: {len(gpus)} GPUs visibles en séquentiel, mesure d'énergie "
                f"limitée au GPU {seq_energy_gpu_index} (--energy-gpu-index pour changer)."
            )
        results = [
            run_variant(
                args.model,
                v,
                args.max_tokens,
                args.gen_tokens,
                compute_ppl=args.compute_ppl,
                ifeval=args.ifeval,
                ifeval_limit=args.ifeval_limit,
                ifeval_batch_size=args.ifeval_batch_size,
                measure_energy=args.measure_energy,
                energy_gpu_index=seq_energy_gpu_index,
                energy_dir=Path(args.energy_out) / v if args.energy_out else None,
            )
            for v in args.variants
        ]

    elapsed = time.time() - t0

    # Réordonner selon l'ordre demandé dans --variants (le parallèle ne le garantit pas).
    order = {v: i for i, v in enumerate(args.variants)}
    results.sort(key=lambda r: order.get(r["variant"], 999))

    table_lines = print_table(results, ifeval=args.ifeval)
    print(f"\ntemps total: {elapsed:.1f}s")

    out_path = (
        Path(args.out)
        if args.out
        else Path("results") / f"{args.model.replace('/', '_')}_{datetime.now():%Y%m%d-%H%M%S}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "variants": args.variants,
                "max_tokens": args.max_tokens,
                "gen_tokens": args.gen_tokens,
                "parallel": parallel,
                "gpus": gpus,
                "compute_ppl": args.compute_ppl,
                "ifeval": args.ifeval,
                "ifeval_limit": args.ifeval_limit,
                "measure_energy": args.measure_energy,
                "elapsed_s": elapsed,
                "results": results,
                "table": "\n".join(table_lines),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"résultats écrits dans {out_path}")


if __name__ == "__main__":
    main()
