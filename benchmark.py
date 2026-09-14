#!/usr/bin/env python
"""Compare un LLM en fp16 / bf16 / int8 / 4bit : VRAM, débit, et en option
perplexité / IFEval / ARC-Challenge.

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
    python benchmark.py --arc --arc-batch-size 16      # + score ARC-Challenge (log-vraisemblance)
    python benchmark.py --arc --arc-chat-template      # ARC avec prompts au format ChatML
    python benchmark.py --ppl                          # + perplexité WikiText-2
    python benchmark.py --variants gptq-int8 gptq-int4 awq --ifeval
        # checkpoints pré-quantifiés hors ligne (repo HF = --model + suffixe,
        # ex: "<model>-GPTQ-Int8") au lieu d'une quantification bitsandbytes
        # à la volée — nécessite que ce checkpoint existe pour --model, et
        # auto-gptq/autoawq installés.
"""
import argparse
import json
import math
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

ALL_VARIANTS = ["fp16", "bf16", "int8", "4bit"]
# Checkpoints déjà quantifiés hors ligne (calibrés une fois, pas à la volée au
# chargement) — contrairement à int8/4bit ci-dessus qui sont quantifiés par
# bitsandbytes au moment du from_pretrained. Suffixe ajouté au nom du modèle
# demandé (--model), suivant la convention Qwen (ex: "Qwen/Qwen2.5-1.5B-
# Instruct" -> "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8"). Nécessite le
# checkpoint correspondant publié pour le modèle choisi, et le backend GPTQ
# (auto-gptq) ou AWQ (autoawq) installé — voir requirements.txt.
PREQUANTIZED_SUFFIXES = {
    "gptq-int8": "-GPTQ-Int8",
    "gptq-int4": "-GPTQ-Int4",
    "awq": "-AWQ",
}
# Défaut : les 4 variantes quantifiées à la volée par bitsandbytes, qui ne
# demandent aucune dépendance supplémentaire. gptq-int8/gptq-int4/awq restent
# utilisables via --variants explicite, mais demandent des versions de
# transformers incompatibles entre elles ET avec celle-ci — chacune dans son
# propre venv, voir requirements-awq.txt et requirements-gptq.txt.
DEFAULT_VARIANTS = ALL_VARIANTS
FORBIDDEN_GPUS = {"3"}  # GPU 3 hors limites sur cette machine, ne jamais l'utiliser.
WARMUP_GEN_TOKENS = 8  # chauffe hors mesure, avant d'entrer dans EnergyMeasurement.
IFEVAL_WARMUP_LIMIT = 1  # idem, pour chauffer le cache dataset IFEval + compiler les kernels.
# Plafond de génération de la tâche IFEval dans lm-evaluation-harness (constaté
# via le warning HF "max_new_tokens (=2048)"). Ne pas y toucher : une réponse
# qui l'atteint n'a pas fini de générer, on ne sait pas ce qu'elle aurait dit
# ensuite — ça ne doit pas être confondu avec une vraie mesure de verbosité.
# À revérifier si la version de lm_eval change.
IFEVAL_GEN_TOKEN_CAP = 2048
# Calibrage --ifeval-batch-size=auto : génération plus courte que le pire cas
# réel (IFEVAL_GEN_TOKEN_CAP) pour rester rapide, marge de sécurité ensuite
# appliquée car les vraies réponses IFEval peuvent remplir un KV-cache bien
# plus gros que ce calibrage.
IFEVAL_CALIBRATION_GEN_TOKENS = 256
IFEVAL_BATCH_SAFETY_MARGIN = 0.5

# ARC-Challenge : split test UNIQUEMENT (ni train ni validation — l'éval est
# 0-shot, il n'y a donc aucun exemple few-shot à tirer). Scoring par
# log-vraisemblance et jamais par génération : pour chaque question on score
# une séquence par option et on prend l'argmax (voir run_arc).
ARC_DATASET_PATH = "allenai/ai2_arc"
ARC_DATASET_NAME = "ARC-Challenge"
ARC_SPLIT = "test"
ARC_TEST_SIZE = 1172  # taille attendue du split test (vérifiée par tests/test_arc_challenge.py)
ARC_PROMPT_TEMPLATE = "Question: {question}\nAnswer:"
# La continuation scorée est le TEXTE de l'option (choices.text), pas la
# lettre A/B/C/D, précédé d'une espace : même convention que le
# target_delimiter de lm-evaluation-harness, pour que les scores restent
# comparables à ceux du harness.
ARC_TARGET_DELIMITER = " "
# Message système FIXÉ pour --arc-chat-template, au lieu du défaut du
# tokenizer : Qwen2.5 base et Instruct n'ont pas le même défaut ("You are a
# helpful assistant." contre "You are Qwen, created by Alibaba Cloud. You are
# a helpful assistant."), ce qui rendrait deux modèles non comparables sur le
# même benchmark. À garder identique entre tous les runs qu'on compare.
ARC_CHAT_SYSTEM_PROMPT = "You are a helpful assistant."
ARC_WARMUP_LIMIT = 1  # chauffe hors mesure, même rôle que IFEVAL_WARMUP_LIMIT.
ARC_DEFAULT_BATCH_SIZE = 16
# Repli si le modèle n'expose pas de longueur de contexte. Les séquences ARC
# font ~100 tokens : la troncature ne se déclenche jamais en pratique.
ARC_FALLBACK_MAX_LENGTH = 2048


def _build_config(variant):
    """Construit la config from_pretrained d'UNE variante — jamais les autres :
    BitsAndBytesConfig(...) vérifie au constructeur que bitsandbytes est
    installé, donc construire toutes les configs par avance casserait tout
    venv qui n'a pas bitsandbytes (ex: .venv-awq, .venv-gptq)."""
    if variant == "fp16":
        return dict(dtype=torch.float16)
    if variant == "bf16":
        return dict(dtype=torch.bfloat16)
    if variant == "int8":
        return dict(dtype=torch.float16, quantization_config=BitsAndBytesConfig(load_in_8bit=True))
    if variant == "4bit":
        return dict(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
        )
    if variant in PREQUANTIZED_SUFFIXES:
        # Checkpoint déjà quantifié sur disque (config.json du repo) : pas de
        # quantization_config à fournir ici.
        return dict(dtype="auto")
    raise KeyError(variant)


def build_configs(selected):
    return {variant: _build_config(variant) for variant in selected}


def resolve_model_name(model_name, variant):
    """Nom du repo HF à charger pour cette variante : le modèle demandé tel
    quel, sauf pour les checkpoints pré-quantifiés qui vivent dans un repo à
    part (voir PREQUANTIZED_SUFFIXES)."""
    suffix = PREQUANTIZED_SUFFIXES.get(variant)
    return f"{model_name}{suffix}" if suffix else model_name


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


def find_max_ifeval_batch_size(model, tok, max_batch=256):
    """Calibre le plus grand batch qui tient dans la VRAM libre, par
    doublement (1, 2, 4, ...), avant d'entrer dans le bloc mesuré. Le
    résultat est ensuite utilisé comme batch EXPLICITE et FIXE pour le run
    réel (décision figée : jamais 'auto' pendant la mesure elle-même, voir
    CLAUDE.md) — seul ce calibrage, hors mesure, est adaptatif."""
    prompt_ids = tok(
        "Write a detailed, step-by-step explanation of how photosynthesis works.",
        return_tensors="pt",
    ).input_ids

    best = 1
    batch = 1
    while batch <= max_batch:
        try:
            torch.cuda.empty_cache()
            ids = prompt_ids.repeat(batch, 1).to(model.device)
            with torch.no_grad():
                model.generate(ids, max_new_tokens=IFEVAL_CALIBRATION_GEN_TOKENS, do_sample=False)
            torch.cuda.synchronize()
            best = batch
            batch *= 2
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break

    torch.cuda.empty_cache()
    # Marge de sécurité : les vraies réponses IFEval peuvent être bien plus
    # longues (jusqu'à IFEVAL_GEN_TOKEN_CAP) que ce calibrage, donc leur
    # KV-cache est plus gros à batch égal.
    return max(1, int(best * IFEVAL_BATCH_SAFETY_MARGIN))


def _parse_ifeval_batch_size(value):
    """Pour --ifeval-batch-size : 'auto' tel quel, sinon un entier."""
    return value if value == "auto" else int(value)


def _extract_metric(metrics, name):
    """lm-eval préfixe les clés avec le nom du filtre (ex: 'prompt_level_strict_acc,none')."""
    for k, v in metrics.items():
        if k.split(",")[0] == name:
            return v
    return None


def run_ifeval(model, tok, limit=None, batch_size=4, log_samples=False):
    """Évalue IFEval (instruction-following) sur le modèle déjà chargé, via
    lm-evaluation-harness. Import différé : lm_eval reste optionnel tant
    qu'on ne passe pas --ifeval.

    Si log_samples, renvoie aussi les échantillons bruts par prompt
    (nécessaires pour compter les tokens générés, voir
    _ifeval_generated_token_counts) — sinon le deuxième élément est None."""
    import logging

    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    logging.getLogger("lm-eval").setLevel(logging.WARNING)
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch_size)
    # apply_chat_template : sans ça, même un modèle Instruct reçoit le prompt
    # en complétion brute au lieu du format conversationnel qu'il attend, et
    # ne sait pas quand s'arrêter (voir le taux de réponses plafonnées).
    out = simple_evaluate(
        model=lm, tasks=["ifeval"], limit=limit, bootstrap_iters=0,
        log_samples=log_samples, apply_chat_template=True,
    )
    samples = out["samples"]["ifeval"] if log_samples else None
    return out["results"]["ifeval"], samples


def _ifeval_generated_token_counts(samples, tok):
    """Longueur (en tokens) de la génération brute pour chaque prompt IFEval.

    Schéma vérifié sur lm-evaluation-harness 0.4.4 : pour une tâche
    generate_until, chaque échantillon a resps=[[texte_généré]] (un seul
    essai, une seule continuation). À revérifier si la version installée
    diffère de celle de requirements.txt."""
    return [len(tok(s["resps"][0][0], add_special_tokens=False).input_ids) for s in samples]


def load_arc_challenge(limit=None):
    """Charge le split test d'ARC-Challenge (ARC_TEST_SIZE items).

    Seul split chargé, volontairement : l'éval est 0-shot, donc ni train ni
    validation ne servent (pas d'exemples few-shot à tirer)."""
    docs = load_dataset(ARC_DATASET_PATH, ARC_DATASET_NAME, split=ARC_SPLIT)
    if limit is not None:
        docs = docs.select(range(min(limit, len(docs))))
    return docs


def arc_gold_index(doc):
    """Index, dans choices.text, de la bonne option.

    answerKey est tantôt une lettre ("A".."E"), tantôt un chiffre ("1".."5")
    selon le document. On ne décode donc jamais la clé : on la cherche dans
    choices.label, qui suit toujours la même convention qu'answerKey au sein
    d'un même document (vérifié sur les 1172 items du split test)."""
    labels = [str(label).strip() for label in doc["choices"]["label"]]
    key = str(doc["answerKey"]).strip()
    if key not in labels:
        raise ValueError(
            f"answerKey {key!r} absent de choices.label {labels} (doc {doc.get('id')!r})."
        )
    return labels.index(key)


def _arc_context(doc):
    return ARC_PROMPT_TEMPLATE.format(question=doc["question"])


def _arc_chat_context(doc, tok):
    """Contexte au format ChatML : la question dans un tour utilisateur, puis
    l'en-tête du tour assistant, que l'option vient compléter.

    Le template se termine par un saut de ligne (ex. "<|im_start|>assistant\n"),
    donc AUCUN délimiteur n'est ajouté avant l'option — contrairement au format
    complétion où une espace sépare "Answer:" du texte."""
    messages = [
        {"role": "system", "content": ARC_CHAT_SYSTEM_PROMPT},
        {"role": "user", "content": doc["question"]},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _encode_arc_pair(tok, context, continuation):
    """(tokens du contexte, tokens de la continuation) pour un modèle causal.

    On encode contexte+continuation d'un bloc puis on coupe à la longueur du
    contexte seul : encoder les deux séparément casserait les fusions de
    tokens à la frontière. Même approche que lm-evaluation-harness."""
    whole = tok(context + continuation, add_special_tokens=False).input_ids
    ctx = tok(context, add_special_tokens=False).input_ids
    return ctx, whole[len(ctx):]


def prepare_arc_requests(tok, docs, chat_template=False):
    """Pré-tokenise toutes les paires (question, option) AVANT le bloc mesuré.

    Une question produit autant de séquences qu'elle a d'options — 3, 4 ou 5
    selon les items, jamais supposé égal à 4.

    Si chat_template, le prompt passe par le template de chat du tokenizer
    (ChatML pour Qwen) au lieu du format complétion — voir _arc_chat_context.
    Les séquences sont alors nettement plus longues (≈ +25 tokens par option
    sur Qwen2.5), ce qui augmente la VRAM du forward à batch égal.

    Séparé de run_arc pour la même raison que la tokenisation de WikiText-2 :
    c'est du travail CPU, le laisser dans le bloc EnergyMeasurement diluerait
    la puissance moyenne avec du temps GPU inactif."""
    if chat_template and getattr(tok, "chat_template", None) is None:
        raise ValueError(
            "--arc-chat-template demandé mais ce tokenizer n'expose aucun "
            "template de chat."
        )
    encoded, byte_lens, n_choices, golds = [], [], [], []
    for doc in docs:
        context = _arc_chat_context(doc, tok) if chat_template else _arc_context(doc)
        delimiter = "" if chat_template else ARC_TARGET_DELIMITER
        texts = doc["choices"]["text"]
        for text in texts:
            encoded.append(_encode_arc_pair(tok, context, delimiter + text))
            # acc_norm normalise par la longueur en OCTETS du texte de l'option,
            # délimiteur et prompt exclus — le format du prompt ne change pas
            # la réponse à normaliser. Pour ne pas favoriser les options
            # courtes (dont
            # la log-vraisemblance, somme de termes négatifs, est mécaniquement
            # plus haute).
            byte_lens.append(len(text.encode("utf-8")))
        n_choices.append(len(texts))
        golds.append(arc_gold_index(doc))
    return {
        "encoded": encoded,
        "byte_lens": byte_lens,
        "n_choices": n_choices,
        "golds": golds,
    }


def _arc_head(requests, n_docs):
    """Les n_docs premières questions d'un jeu de requêtes déjà préparé."""
    n_choices = requests["n_choices"][:n_docs]
    n_seq = sum(n_choices)
    return {
        "encoded": requests["encoded"][:n_seq],
        "byte_lens": requests["byte_lens"][:n_seq],
        "n_choices": n_choices,
        "golds": requests["golds"][:n_docs],
    }


@torch.no_grad()
def _arc_loglikelihoods(model, tok, encoded, batch_size, max_length):
    """Log-vraisemblance totale de chaque continuation sachant son contexte.

    Un forward par batch, de taille FIXE (jamais adaptée à la VRAM : c'est une
    variable expérimentale, voir --arc-batch-size). Les séquences sont
    traitées dans l'ordre du dataset, sans tri par longueur, pour que le
    nombre et la forme des forwards soient identiques d'une variante à
    l'autre.

    Renvoie (log-vraisemblances, nombre de forwards effectués)."""
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    lls = []
    forward_passes = 0

    for start in range(0, len(encoded), batch_size):
        chunk = encoded[start : start + batch_size]
        # Le dernier token n'a pas de cible : on donne au modèle ctx+cont
        # amputé de son dernier token et on lit la distribution prédite pour
        # chaque token de la continuation.
        seqs = [(ctx + cont)[-max_length:][:-1] for ctx, cont in chunk]
        width = max(len(s) for s in seqs)
        # Padding à DROITE : l'attention causale empêche les vrais tokens, tous
        # alignés à gauche, de voir le padding — les logits qu'on lit sont donc
        # identiques à ceux d'un forward sans padding.
        input_ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, seq in enumerate(seqs):
            input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attn[i, : len(seq)] = 1

        logits = model(
            input_ids=input_ids.to(model.device), attention_mask=attn.to(model.device)
        ).logits
        forward_passes += 1

        for i, (_ctx, cont) in enumerate(chunk):
            # Les len(cont) dernières positions de seqs[i] prédisent exactement
            # les tokens de la continuation.
            end = len(seqs[i])
            # log_softmax sur la seule tranche utile : le faire sur tout le
            # batch coûterait un tenseur float32 de taille batch x width x vocab.
            logprobs = torch.log_softmax(logits[i, end - len(cont) : end, :].float(), dim=-1)
            targets = torch.tensor(cont, dtype=torch.long, device=logprobs.device)
            lls.append(logprobs.gather(-1, targets.unsqueeze(-1)).sum().item())

    return lls, forward_passes


def _binomial_stderr(p, n):
    """Erreur-type binomiale d'une proportion mesurée sur n items."""
    return math.sqrt(p * (1.0 - p) / n) if n > 0 else None


def run_arc(model, tok, requests, batch_size=ARC_DEFAULT_BATCH_SIZE):
    """Évalue ARC-Challenge sur le modèle déjà chargé, par log-vraisemblance.

    `requests` vient de prepare_arc_requests : tout est déjà tokenisé, aucune
    I/O ici, la fonction est faite pour tourner dans le bloc mesuré par
    EnergyMeasurement.

    Deux métriques, l'argmax étant pris sur les options de CETTE question
    (3, 4 ou 5, jamais supposé) :
    - acc      : argmax de la log-vraisemblance brute ;
    - acc_norm : argmax de la log-vraisemblance divisée par la longueur en
                 octets du texte de l'option — c'est la métrique principale.

    Renvoie un dict de métriques, enrichi du nombre de forwards effectués
    (pour la normalisation énergétique)."""
    max_length = (
        getattr(model.config, "max_position_embeddings", None) or ARC_FALLBACK_MAX_LENGTH
    )
    lls, forward_passes = _arc_loglikelihoods(
        model, tok, requests["encoded"], batch_size, max_length
    )

    byte_lens, golds = requests["byte_lens"], requests["golds"]
    correct = correct_norm = 0
    offset = 0
    for gold, k in zip(golds, requests["n_choices"]):
        window = lls[offset : offset + k]
        normed = [ll / b for ll, b in zip(window, byte_lens[offset : offset + k])]
        correct += int(max(range(k), key=window.__getitem__) == gold)
        correct_norm += int(max(range(k), key=normed.__getitem__) == gold)
        offset += k

    n = len(golds)
    acc, acc_norm = correct / n, correct_norm / n
    return {
        "acc": acc,
        "acc_stderr": _binomial_stderr(acc, n),
        "acc_norm": acc_norm,
        "acc_norm_stderr": _binomial_stderr(acc_norm, n),
        "n": n,
        "sequences_scored": len(lls),
        "forward_passes": forward_passes,
    }


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
    arc=False,
    arc_limit=None,
    arc_batch_size=ARC_DEFAULT_BATCH_SIZE,
    arc_chat_template=False,
    measure_energy=True,
    energy_gpu_index=0,
    energy_dir=None,
):
    """Charge tokenizer+modèle et évalue UNE variante. Suppose que
    CUDA_VISIBLE_DEVICES est déjà positionné correctement par l'appelant.

    Si measure_energy est activé, le débit, la perplexité (si demandée),
    IFEval et ARC-Challenge (si demandés) tournent tous sous
    EnergyMeasurement, sur `energy_gpu_index` (index PHYSIQUE nvidia-smi du
    GPU réellement utilisé par ce process — voir le README de
    energy_measurement/). Comme
    lm-evaluation-harness télécharge/charge son dataset et compile des
    kernels de génération au premier appel, un passage IFEval "à vide" (un
    seul exemple) est fait hors mesure juste avant, uniquement pour chauffer
    ce cache — le run IFEval réel (celui compté dans les résultats) a lieu
    dans le bloc mesuré. Même principe pour ARC-Challenge : dataset chargé et
    intégralement tokenisé hors mesure, puis une question de chauffe, et seuls
    les forwards du scoring réel tombent dans le bloc mesuré."""
    # Pour un checkpoint pré-quantifié (gptq-int8/gptq-int4/awq), load_name
    # pointe vers son propre repo HF ; model_name reste le nom "logique"
    # demandé, utilisé pour les métadonnées et les dossiers de résultats afin
    # que toutes les variantes d'un même modèle restent groupées ensemble.
    load_name = resolve_model_name(model_name, variant)
    tok = AutoTokenizer.from_pretrained(load_name)

    cfg = build_configs([variant])[variant]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(load_name, device_map="cuda", **cfg)
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

    arc_requests = None
    if arc:
        arc_requests = prepare_arc_requests(
            tok, load_arc_challenge(limit=arc_limit), chat_template=arc_chat_template
        )

    # Chauffe hors mesure : le premier appel au modèle compile des kernels et
    # alloue de la mémoire, ce qui fausserait aussi bien tok/s que la trace
    # de puissance si on le laissait dans le bloc mesuré.
    throughput(model, tok, WARMUP_GEN_TOKENS)
    if ifeval:
        if ifeval_batch_size == "auto":
            # Calibrage hors mesure : remplace "auto" par le plus grand batch
            # explicite qui tient en VRAM pour CE modèle/CETTE variante.
            ifeval_batch_size = find_max_ifeval_batch_size(model, tok)
            torch.cuda.reset_peak_memory_stats()  # le calibrage ne doit pas polluer le pic VRAM rapporté
        # Idem pour IFEval : premier appel = téléchargement/chargement du
        # dataset + compilation. Un seul exemple suffit à chauffer le cache
        # (le dataset entier est mis en cache local dès ce premier appel).
        run_ifeval(model, tok, limit=IFEVAL_WARMUP_LIMIT, batch_size=ifeval_batch_size)  # (metrics, None)
    if arc:
        # Idem : le premier forward compile des kernels. Une seule question
        # (ARC_WARMUP_LIMIT) suffit, le dataset est déjà tokenisé au-dessus.
        run_arc(model, tok, _arc_head(arc_requests, ARC_WARMUP_LIMIT), batch_size=arc_batch_size)
    torch.cuda.synchronize()

    def _run_ifeval_and_record(res):
        t0 = time.time()
        ifeval_metrics, samples = run_ifeval(
            model, tok, limit=ifeval_limit, batch_size=ifeval_batch_size, log_samples=True
        )
        ifeval_elapsed_s = time.time() - t0
        res["ifeval"] = ifeval_metrics
        res["ifeval_score"] = _extract_metric(ifeval_metrics, "prompt_level_strict_acc")
        # Batch réellement utilisé (fixe, explicite) — y compris quand il vient
        # du calibrage auto, pour garder une trace exacte (décision figée #5).
        res["ifeval_batch_size_used"] = ifeval_batch_size

        token_counts = _ifeval_generated_token_counts(samples, tok)
        total_tokens = sum(token_counts)
        res["ifeval_tokens_per_response"] = token_counts
        res["ifeval_total_tokens"] = total_tokens
        res["ifeval_mean_tokens_per_response"] = total_tokens / len(token_counts)
        capped = sum(1 for c in token_counts if c >= IFEVAL_GEN_TOKEN_CAP)
        res["ifeval_capped_responses"] = capped
        res["ifeval_capped_rate"] = capped / len(token_counts)
        res["ifeval_elapsed_s"] = ifeval_elapsed_s
        res["ifeval_time_per_token_s"] = ifeval_elapsed_s / total_tokens

    def _run_arc_and_record(res):
        t0 = time.time()
        arc_metrics = run_arc(model, tok, arc_requests, batch_size=arc_batch_size)
        arc_elapsed_s = time.time() - t0
        res["arc"] = arc_metrics
        # Métrique principale d'ARC-Challenge : acc_norm (log-vraisemblance
        # normalisée par la longueur de l'option), pas l'accuracy brute.
        res["arc_score"] = arc_metrics["acc_norm"]
        res["arc_acc"] = arc_metrics["acc"]
        res["arc_acc_norm"] = arc_metrics["acc_norm"]
        # Batch fixe, jamais calibré automatiquement, contrairement à IFEval :
        # c'est une variable expérimentale (décision figée), elle doit rester
        # identique d'une variante à l'autre pour que la comparaison tienne.
        res["arc_batch_size_used"] = arc_batch_size
        # Format du prompt : complétion ou ChatML. Change les scores autant que
        # la quantization, donc jamais comparer deux runs qui diffèrent dessus.
        res["arc_chat_template"] = arc_chat_template
        # Unité de travail GPU d'ARC (le scoring ne génère rien) : sert à
        # normaliser l'énergie, comme les tokens générés pour IFEval.
        res["arc_forward_passes"] = arc_metrics["forward_passes"]
        res["arc_sequences_scored"] = arc_metrics["sequences_scored"]
        res["arc_elapsed_s"] = arc_elapsed_s
        res["arc_time_per_forward_s"] = arc_elapsed_s / arc_metrics["forward_passes"]

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
            if arc:
                _run_arc_and_record(result)
        result["energy_j"] = em.energy_j
        result["energy_wh"] = em.energy_j / 3600.0
        result["mean_power_w"] = em.mean_power_w
        result["mean_utilization_pct"] = em.mean_utilization_pct
        result["peak_vram_mib"] = em.peak_vram_mib
        result["mean_vram_mib"] = em.mean_vram_mib
        result["energy_run_dir"] = str(em.run_dir)
        if result.get("ifeval_total_tokens"):
            # Approximation : le bloc mesuré inclut aussi le probe tok/s
            # (WARMUP_GEN_TOKENS tokens, négligeable face aux ~541 prompts
            # IFEval) — attribué ici entièrement à IFEval plutôt que réparti.
            result["energy_per_token_j"] = result["energy_j"] / result["ifeval_total_tokens"]
        if result.get("arc_forward_passes"):
            # Même approximation : toute l'énergie du bloc est attribuée à ARC.
            # Si --ifeval et --arc tournent ensemble, les deux ratios comptent
            # chacun l'énergie totale — ils ne sont alors pas additifs, et il
            # faut un run par benchmark pour un coût énergétique par tâche net.
            result["energy_per_forward_pass_j"] = (
                result["energy_j"] / result["arc_forward_passes"]
            )
    else:
        result["tok_s"] = throughput(model, tok, gen_tokens)
        if compute_ppl:
            result["ppl"] = perplexity(model, ppl_enc)
        if ifeval:
            _run_ifeval_and_record(result)
        if arc:
            _run_arc_and_record(result)

    # Mesuré en dernier : capture le pic mémoire de toute l'évaluation
    # (ppl + tok/s + ifeval + arc).
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
    arc=False,
    arc_limit=None,
    arc_batch_size=ARC_DEFAULT_BATCH_SIZE,
    arc_chat_template=False,
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
    if arc:
        cmd.append("--arc")
        if arc_limit is not None:
            cmd += ["--arc-limit", str(arc_limit)]
        cmd += ["--arc-batch-size", str(arc_batch_size)]
        if arc_chat_template:
            cmd.append("--arc-chat-template")
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
    if result.get("arc_score") is not None:
        extra += f" arc={result['arc_score'] * 100:.1f}%"
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
    arc=False,
    arc_limit=None,
    arc_batch_size=ARC_DEFAULT_BATCH_SIZE,
    arc_chat_template=False,
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
                arc=arc,
                arc_limit=arc_limit,
                arc_batch_size=arc_batch_size,
                arc_chat_template=arc_chat_template,
                measure_energy=measure_energy,
                energy_out=energy_out,
            )
        finally:
            gpu_queue.put(gpu_id)

    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        return list(ex.map(worker, variants))


def print_table(results, ifeval=False, arc=False):
    header = f"\n{'variant':8} {'ppl':>9} {'VRAM_GB':>9} {'tok/s':>8} {'energy_Wh':>10} {'avg_W':>7}"
    if ifeval:
        header += f" {'ifeval':>8}"
    if arc:
        header += f" {'arc_norm':>9}"
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
            if arc:
                score = r.get("arc_score")
                line += f" {score * 100:8.1f}%" if score is not None else f" {'n/a':>9}"
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
        default=DEFAULT_VARIANTS,
        choices=ALL_VARIANTS + list(PREQUANTIZED_SUFFIXES),
        help="Parmi fp16/bf16/int8/4bit (quantifiés à la volée par bitsandbytes), "
        "ou gptq-int8/gptq-int4/awq (checkpoints déjà quantifiés hors ligne, "
        "repo HF = --model + suffixe, ex: '<model>-GPTQ-Int8' — nécessite que "
        "ce checkpoint existe pour le modèle choisi, et auto-gptq/autoawq installé).",
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
        type=_parse_ifeval_batch_size,
        default="auto",
        help="Batch size pour l'évaluation IFEval : un entier fixe, ou 'auto' "
        "(défaut) pour calibrer automatiquement le plus grand batch qui tient "
        "dans la VRAM libre, hors mesure, avant de l'utiliser comme batch fixe "
        "pour le run réel (voir find_max_ifeval_batch_size).",
    )
    ap.add_argument(
        "--arc",
        action="store_true",
        help="Évalue aussi ARC-Challenge (raisonnement scientifique, split test, "
        "1172 questions) par log-vraisemblance — pas de génération.",
    )
    ap.add_argument(
        "--arc-limit",
        type=int,
        default=None,
        help="Limite le nombre de questions ARC (défaut: tout le split test, 1172).",
    )
    ap.add_argument(
        "--arc-batch-size",
        type=int,
        default=ARC_DEFAULT_BATCH_SIZE,
        help=f"Nombre de séquences scorées par forward pour ARC (défaut: "
        f"{ARC_DEFAULT_BATCH_SIZE}). Entier FIXE, jamais 'auto' contrairement à "
        "--ifeval-batch-size : c'est une variable expérimentale, garde-la "
        "identique entre variantes pour que les comparaisons tiennent.",
    )
    ap.add_argument(
        "--arc-chat-template",
        action="store_true",
        help="Formate les prompts ARC avec le template de chat du tokenizer "
        "(ChatML pour Qwen) au lieu du format complétion 'Question: ...\\nAnswer:'. "
        "Pensé pour un modèle instruct. Attention : allonge les séquences "
        "(≈ +25 tokens/option sur Qwen2.5), donc augmente la VRAM à batch égal, "
        "et les scores ne sont PAS comparables à ceux d'un run en complétion.",
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
            arc=args.arc,
            arc_limit=args.arc_limit,
            arc_batch_size=args.arc_batch_size,
            arc_chat_template=args.arc_chat_template,
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
                arc=args.arc,
                arc_limit=args.arc_limit,
                arc_batch_size=args.arc_batch_size,
                arc_chat_template=args.arc_chat_template,
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
                arc=args.arc,
                arc_limit=args.arc_limit,
                arc_batch_size=args.arc_batch_size,
                arc_chat_template=args.arc_chat_template,
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

    table_lines = print_table(results, ifeval=args.ifeval, arc=args.arc)
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
                "arc": args.arc,
                "arc_limit": args.arc_limit,
                "arc_batch_size": args.arc_batch_size,
                "arc_chat_template": args.arc_chat_template,
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
