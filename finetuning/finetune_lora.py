#!/usr/bin/env python
"""LoRA sur Qwen2.5-1.5B quantifié en 4bit (QLoRA), pour le suivi d'instructions.

Dataset : allenai/tulu-3-sft-personas-instruction-following (sous-ensemble
"instruction following" du mélange SFT de Tulu 3), au format `messages`.

Recette :
  - base gelée en 4bit NF4 + double quantification, calcul en bf16 (A100)
  - LoRA r=16 sur les projections attention + MLP
  - perte calculée UNIQUEMENT sur la réponse de l'assistant (le prompt est
    masqué à -100), ce qui est ce qu'on veut pour de l'instruction tuning

L'objectif est d'évaluer ensuite sur IFEval : voir eval_ifeval.py dans ce
dossier. Attention au formatage — on entraîne avec le chat template, donc
il faut aussi l'appliquer à l'évaluation (eval_ifeval.py le fait par défaut
quand un adaptateur est chargé).

Usage:
    python finetuning/finetune_lora.py
    python finetuning/finetune_lora.py --max-samples 2000 --epochs 1   # essai rapide
    CUDA_VISIBLE_DEVICES=0 python finetuning/finetune_lora.py
"""
import argparse
import os

# Aligne la numérotation CUDA sur celle de nvidia-smi, comme benchmark.py.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# Projections ciblées par LoRA sur l'architecture Qwen2 (attention + MLP).
QWEN_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def build_tokenizer(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        # Qwen n'a pas toujours de pad token distinct ; l'eos fait l'affaire,
        # les positions paddées sont de toute façon masquées (-100 / attention).
        tok.pad_token = tok.eos_token
    if tok.chat_template is None:
        raise SystemExit(
            f"{model_name} n'expose pas de chat_template. Ce script formate les "
            "exemples avec apply_chat_template ; choisis un modèle qui en a un, "
            "ou définis tok.chat_template manuellement."
        )
    return tok


def build_dataset(tok, dataset_name, split, max_seq_len, max_samples, seed, num_proc):
    """Tokenise le dataset au format chat, en masquant le prompt dans les labels.

    Chaque exemple a une clé `messages` (liste de {role, content}). On construit :
      - prompt_text : tous les messages sauf la réponse finale, + amorce assistant
      - full_text   : la conversation complète
    prompt_text est un préfixe strict de full_text, donc masquer les N premiers
    tokens (N = len(prompt_ids)) revient à ne calculer la perte que sur la réponse.
    """
    ds = load_dataset(dataset_name, split=split)
    if max_samples:
        ds = ds.shuffle(seed=seed).select(range(min(max_samples, len(ds))))

    def encode(example):
        messages = example["messages"]
        prompt_text = tok.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        full_text = tok.apply_chat_template(messages, tokenize=False)

        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        input_ids = tok(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_len,
        )["input_ids"]

        labels = list(input_ids)
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100
        return {"input_ids": input_ids, "labels": labels}

    encoded = ds.map(
        encode,
        remove_columns=ds.column_names,
        num_proc=num_proc,
        desc="tokenisation",
    )
    # Un exemple tronqué au point de n'avoir que du prompt n'apporte aucun signal.
    keep = encoded.filter(
        lambda ex: any(label != -100 for label in ex["labels"]),
        num_proc=num_proc,
        desc="filtrage des exemples sans réponse",
    )
    dropped = len(encoded) - len(keep)
    if dropped:
        print(f"{dropped} exemple(s) écarté(s) : réponse hors fenêtre après troncature.")
    return keep


def build_model(model_name, lora_r, lora_alpha, lora_dropout, grad_checkpointing):
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # NF4 = recette QLoRA (≠ fp4, défaut de bnb)
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_cfg,
        dtype=torch.bfloat16,
        device_map={"": 0},  # un seul GPU : pas de sharding
    )
    model.config.use_cache = False  # incompatible avec le gradient checkpointing
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=grad_checkpointing
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=QWEN_LORA_TARGETS,
        ),
    )
    model.print_trainable_parameters()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument(
        "--dataset", default="allenai/tulu-3-sft-personas-instruction-following"
    )
    ap.add_argument("--split", default="train")
    ap.add_argument("--output-dir", default="finetuning/out/qwen2.5-1.5b-qlora-ifeval")
    ap.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limite le nombre d'exemples (0 = tout le split). Utile pour un essai rapide.",
    )
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-proc", type=int, default=8, help="Process pour la tokenisation.")
    ap.add_argument(
        "--no-grad-checkpointing",
        dest="grad_checkpointing",
        action="store_false",
        help="Désactive le gradient checkpointing (plus rapide, plus gourmand en VRAM).",
    )
    ap.set_defaults(grad_checkpointing=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA requis (bitsandbytes ne quantifie que sur GPU).")

    print(f"modèle  = {args.model}")
    print(f"dataset = {args.dataset} (split {args.split})")
    print(f"device  = {torch.cuda.get_device_name()}")

    tok = build_tokenizer(args.model)
    train_ds = build_dataset(
        tok,
        args.dataset,
        args.split,
        args.max_seq_len,
        args.max_samples,
        args.seed,
        args.num_proc,
    )
    print(f"{len(train_ds)} exemples d'entraînement")

    model = build_model(
        args.model,
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
        args.grad_checkpointing,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,  # A100 : bf16 natif, et c'est le dtype d'entraînement de Qwen
        gradient_checkpointing=args.grad_checkpointing,
        # use_reentrant=False : requis pour que le checkpointing coopère avec PEFT
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",  # optimiseur paginé bitsandbytes (recette QLoRA)
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tok,
        data_collator=DataCollatorForSeq2Seq(
            tok, padding=True, label_pad_token_id=-100
        ),
    )
    trainer.train()

    # On ne sauvegarde que l'adaptateur LoRA (quelques dizaines de Mo) : la base
    # 4bit est rechargée depuis le Hub à l'évaluation.
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"\nadaptateur LoRA écrit dans {args.output_dir}")
    print(
        "Évaluation :\n"
        f"    python finetuning/eval_ifeval.py --adapter {args.output_dir}"
    )


if __name__ == "__main__":
    main()
