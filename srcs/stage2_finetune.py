import os
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

import transformers
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    modeling_utils,
)
from torch.utils.data import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
)

if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or \
        modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]

IGNORE_INDEX = -100
MAX_LENGTH   = 2048

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


class Stage2Dataset(Dataset):
    def __init__(self, csv_path: str, tokenizer, max_length: int = MAX_LENGTH):
        super().__init__()
        df = pd.read_csv(csv_path)
        self.samples   = df.to_dict(orient="records")
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"[Stage2Dataset] Loaded {len(self.samples)} samples")

        if "perturb_type" in df.columns:
            counts = df["perturb_type"].value_counts()
            print(f"  continuous: {counts.get('continuous', 0)}")
            print(f"  composition: {counts.get('composition', 0)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_text  = sample["input_text"]
        target_text = sample["target_text"]

        full_text = input_text + "\n" + target_text + self.tokenizer.eos_token

        full_tokens = self.tokenizer(
            full_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        )
        input_ids = full_tokens.input_ids[0]

        input_only_tokens = self.tokenizer(
            input_text + "\n",
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        )
        input_len = input_only_tokens.input_ids.shape[1]

        labels = input_ids.clone()
        labels[:input_len] = IGNORE_INDEX

        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids.shape[0],
        )


@dataclass
class Stage2DataCollator:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids = [inst["input_ids"].clone().detach()
                     for inst in instances]
        labels    = [inst["labels"].clone().detach()
                     for inst in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def load_model_and_tokenizer(args):
    print(f"\n[Stage2] Loading base model: {args.base_model_path}")
    model = LlamaForCausalLM.from_pretrained(
        args.base_model_path,
        load_in_8bit=True,
        device_map="auto",
        local_files_only=True,
    )

    print(f"[Stage2] Loading tokenizer: {args.stage1_lora_path}")
    tokenizer = LlamaTokenizer.from_pretrained(
        args.stage1_lora_path,
        model_max_length=MAX_LENGTH,
        padding_side="right",
        use_fast=False,
        local_files_only=True,
    )

    model.resize_token_embeddings(len(tokenizer))
    print(f"[Stage2] Tokenizer vocabulary size: {len(tokenizer)}")

    print(f"[Stage2] Loading Stage 1 LoRA weights: {args.stage1_lora_path}")
    model = PeftModel.from_pretrained(
        model,
        args.stage1_lora_path,
        is_trainable=True,
    )

    print(f"[Stage2] Adding Stage 2 LoRA adapter on top of Stage 1 LoRA")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.add_adapter("stage2", lora_config)
    model.set_adapter("stage2")
    model.print_trainable_parameters()

    return model, tokenizer


def setup_training_args(args):
    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["WANDB_DISABLED"] = "True"
    os.environ["ACCELERATE_MIXED_PRECISION"] = "no"

    return TrainingArguments(
        output_dir=str(output_dir),
        run_name=args.run_name,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=0.0,
        fp16=True,
        bf16=False,
        gradient_checkpointing=False,
        ddp_find_unused_parameters=False,
        eval_strategy="steps",
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=10,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        label_names=["labels"],
        fsdp=[],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
    )


def main(args):
    print("=" * 60)
    print("IRLCrys Stage 2: Correction-Oriented Fine-tuning")
    print("=" * 60)
    print(f"  base_model:      {args.base_model_path}")
    print(f"  stage1_lora:     {args.stage1_lora_path}")
    print(f"  data:            {args.data_path}")
    print(f"  output_dir:      {args.output_dir}/{args.run_name}")
    print(f"  lr:              {args.lr}")
    print(f"  epochs:          {args.num_epochs}")
    print(f"  batch_size:      {args.batch_size}")
    print(f"  grad_accum:      {args.grad_accum}")
    print(f"  effective_batch: {args.batch_size * args.grad_accum}")
    print("=" * 60)

    model, tokenizer = load_model_and_tokenizer(args)

    df = pd.read_csv(args.data_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    if args.max_samples > 0 and len(df) > args.max_samples:
        df = df.iloc[:args.max_samples]
        print(f"[Stage2] Randomly sampled {args.max_samples} samples (original total: {len(pd.read_csv(args.data_path))})")
    n_val = max(10, int(len(df) * 0.05))
    df_val   = df.iloc[:n_val]
    df_train = df.iloc[n_val:]

    tmp_dir = Path(args.output_dir) / "tmp_splits"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    train_csv = str(tmp_dir / "stage2_split_train.csv")
    val_csv   = str(tmp_dir / "stage2_split_val.csv")
    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv,   index=False)

    print(f"\n[Stage2] Training samples: {len(df_train)}, Validation samples: {len(df_val)}")

    train_dataset = Stage2Dataset(train_csv, tokenizer)
    val_dataset   = Stage2Dataset(val_csv,   tokenizer)

    data_collator = Stage2DataCollator(tokenizer=tokenizer)

    training_args = setup_training_args(args)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.001,
        )],
    )

    print("\n[Stage2] Starting training...")
    train_result = trainer.train()
    print(f"\n[Stage2] Training complete: {train_result}")

    trainer.save_state()
    trainer.save_model()
    print(f"\n[Stage2] Model saved to: {args.output_dir}/{args.run_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path",
                        type=str,
                        default="/home/wx/MatLLM/Llama-2-7b-hf",
                        help="LLaMA-2 base model path")
    parser.add_argument("--stage1_lora_path",
                        type=str,
                        default="/home/wx/MatLLM/IRLCrys/exp/7b-mp-attr/checkpoint-27136",
                        help="Stage 1 fine-tuned LoRA weight path")
    parser.add_argument("--data_path",
                        type=str,
                        default="/home/wx/MatLLM/IRLCrys/data/mp_20/stage2_train.csv",
                        help="Stage 2 training data CSV path")
    parser.add_argument("--output_dir",
                        type=str,
                        default="/home/wx/MatLLM/IRLCrys/exp",
                        help="Output directory")
    parser.add_argument("--run_name",
                        type=str,
                        default="stage2-7b-mp",
                        help="Experiment name")
    parser.add_argument("--num_epochs",
                        type=int,
                        default=5)
    parser.add_argument("--batch_size",
                        type=int,
                        default=1)
    parser.add_argument("--grad_accum",
                        type=int,
                        default=4,
                        help="Gradient accumulation steps, effective_batch = batch_size x grad_accum")
    parser.add_argument("--lr",
                        type=float,
                        default=5e-5,
                        help="Stage 2 learning rate (lower than Stage 1's 1e-4)")
    parser.add_argument("--lora_rank",
                        type=int,
                        default=8)
    parser.add_argument("--lora_alpha",
                        type=int,
                        default=32)
    parser.add_argument("--lora_dropout",
                        type=float,
                        default=0.05)
    parser.add_argument("--eval_freq",
                        type=int,
                        default=500)
    parser.add_argument("--save_freq",
                        type=int,
                        default=1000)
    parser.add_argument("--max_samples",
                        type=int,
                        default=20000,
                        help="Maximum number of samples to randomly sample, 0 means use all data")
    args = parser.parse_args()
    main(args)