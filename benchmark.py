#!/usr/bin/env python
"""Compare un LLM en fp32 / fp16 / int8 / 4bit : VRAM, débit, et en option
perplexité / IFEval.

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

ALL_VARIANTS = ["fp32", "fp16", "int8", "4bit"]
FORBIDDEN_GPUS = {"3"}  # GPU 3 hors limites sur cette machine, ne jamais l'utiliser.


def build_configs(selected):
    all_cfg = {
        "fp32": dict(dtype=torch.float32),
        "fp16": dict(dtype=torch.float16),
        "int8": dict(
            dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        ),
        "4bit": dict(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
        ),
    }
    return {k: all_cfg[k] for k in selected}


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


def _extract_metric(metrics, name):
    """lm-eval préfixe les clés avec le nom du filtre (ex: 'prompt_level_strict_acc,none')."""
    for k, v in metrics.items():
        if k.split(",")[0] == name:
            return v
    return None


def run_ifeval(model, tok, limit=None, batch_size=4):
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


def run_variant(
    model_name,
    variant,
    max_tokens,
    gen_tokens,
    compute_ppl=False,
    ifeval=False,
    ifeval_limit=None,
    ifeval_batch_size=4,
):
    """Charge tokenizer+modèle et évalue UNE variante. Suppose que
    CUDA_VISIBLE_DEVICES est déjà positionné correctement par l'appelant."""
    tok = AutoTokenizer.from_pretrained(model_name)

    cfg = build_configs([variant])[variant]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", **cfg)
    model.eval()

    tps = throughput(model, tok, gen_tokens)

    result = {"variant": variant, "tok_s": tps}

    if compute_ppl:
        text = "\n\n".join(
            load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
        )
        enc = tok(text, return_tensors="pt")
        if max_tokens:
            enc.input_ids = enc.input_ids[:, :max_tokens]
        result["ppl"] = perplexity(model, enc)

    if ifeval:
        ifeval_metrics = run_ifeval(model, tok, limit=ifeval_limit, batch_size=ifeval_batch_size)
        result["ifeval"] = ifeval_metrics
        result["ifeval_score"] = _extract_metric(ifeval_metrics, "prompt_level_strict_acc")

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
    ifeval_batch_size=4,
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
    ]
    if compute_ppl:
        cmd.append("--ppl")
    if ifeval:
        cmd.append("--ifeval")
        if ifeval_limit is not None:
            cmd += ["--ifeval-limit", str(ifeval_limit)]
        cmd += ["--ifeval-batch-size", str(ifeval_batch_size)]

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
    ifeval_batch_size=4,
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
            )
        finally:
            gpu_queue.put(gpu_id)

    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        return list(ex.map(worker, variants))


def print_table(results, ifeval=False):
    header = f"\n{'variant':8} {'ppl':>9} {'VRAM_GB':>9} {'tok/s':>8}"
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
            line = f"{r['variant']:8} {ppl_str} {r['vram_gb']:9.2f} {r['tok_s']:8.1f}"
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
        type=int,
        default=4,
        help="Batch size pour l'évaluation IFEval.",
    )
    # Flags internes utilisés par les sous-process workers, cachés du --help.
    ap.add_argument("--worker-variant", dest="worker_variant", help=argparse.SUPPRESS)
    ap.add_argument("--result-file", dest="result_file", help=argparse.SUPPRESS)
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
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        print(f"device = {torch.cuda.get_device_name()}   (exécution séquentielle)")
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
