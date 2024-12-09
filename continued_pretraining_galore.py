import os
import random
import argparse
import math
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from trl import SFTTrainer
from datasets  import Dataset, load_dataset, concatenate_datasets
from transformers import get_linear_schedule_with_warmup
from unsloth import FastLanguageModel, is_bfloat16_supported, UnslothTrainingArguments

from galore_torch import GaLoreAdamW8bit
from train_utils import formatCPTV1, formatCPTV2, formatCPTV3


CPT_LOCAL_DATA = [
    ("./dataset/HAM_GPT_dataset_v1119.csv", formatCPTV1),
    ("./dataset/K_IFRS_GPT_QA_v1122.csv", formatCPTV1),
    ("./dataset/economics_fewshot_GPT_dataset.csv", formatCPTV1),
    ("./dataset/math_HAM_GPT_dataset.csv", formatCPTV1),
    ("./dataset/math_fewshot_GPT_dataset.csv", formatCPTV1),
    ("./dataset/haerae_accounting_fewshot_GPT_dataset.csv", formatCPTV1),
    ("./dataset/haerae_eocnomics_fewshot_GPT_dataset.csv", formatCPTV1),
    ("./dataset/K-IFRS-v3.csv", formatCPTV1),
    ("./dataset/CFA-v1.csv", formatCPTV1),
    ("./dataset/CFA-QA-v1.csv", formatCPTV1),
    ("./dataset/sample_K_IFRS.csv", formatCPTV1),
    ("./dataset/QA_from_pdfs.csv", formatCPTV1),
    ("./dataset/sample_accounting_gosi_v3.csv", formatCPTV3),
    ("./dataset/KRX-crawling.csv", formatCPTV1),
    ("./dataset/KRX_text_QA_final.csv", formatCPTV2),
    ("./dataset/dataset_cfa_program_fundamentals_ebook.csv", formatCPTV1),
    ("./dataset/dataset_International_Financial_Markets_and_Monetary_Policy.csv", formatCPTV1),
    ("./dataset/dataset_introduction_to_financial_analysis.csv", formatCPTV1),
    ("./dataset/dataset_principles_of_financial_accounting.csv", formatCPTV1),
    ("./dataset/dataset_Review_Papers_for_Journal_of_Risk_and_Financial_Management_JRFM.csv", formatCPTV1),
    ("./dataset/dataset_principles_of_finance.csv", formatCPTV1),
    ("./dataset/dataset_principles_of_financial_accounting_volume_1_financial_accounting_v2.csv", formatCPTV1)
]

CPT_HF_DATA = [
    ("Cartinoe5930/raw_text_synthetic_dataset_50k", formatCPTV1),
    ("Cartinoe5930/web_text_synthetic_dataset_50k", formatCPTV1),
    ("amphora/krx-sample-instructions", formatCPTV1)
]


def parse_args():
    parser = argparse.ArgumentParser()
    
    # model & tokenizer setting
    parser.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max_token_length", default=2048, type=int)
    parser.add_argument("--use_cache", action='store_true')

    # galore setting
    parser.add_argument("--galore_rank", default=128, type=int)
    parser.add_argument("--galore_update_proj_gap", default=200, type=int)
    parser.add_argument("--galore_scale", default=0.0, type=float)
    parser.add_argument("--galore_proj_type", default="std", type=str)

    # training arguments setting
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--embedding_lr", default=1e-5, type=float)
    parser.add_argument("--lr_warmup_ratio", default=0.06, type=float)
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument("--max_grad_norm", default=2.0, type=float)
    parser.add_argument("--use_gradient_checkpointing", action='store_true')
    
    args = parser.parse_args()
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    # arguments
    args = parse_args()
    
    # set seed
    set_seed(args.seed)

    # load model & tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        dtype=None,
        load_in_4bit=False,
        max_seq_length=args.max_token_length,
        device_map="auto"
    )
    model.config.use_cache = args.use_cache
    tokenizer.padding_side = "left"

    # load and preprocess dataset
    dataset = Dataset.from_dict({})

    # local dataset
    for path, preprocess_fn in CPT_LOCAL_DATA:
        raw_dataset = Dataset.from_pandas(pd.read_csv(path))
        preprocessed_dataset = raw_dataset.map(preprocess_fn, remove_columns=raw_dataset.column_names)
        dataset = concatenate_datasets([dataset, preprocessed_dataset])

    # huggingface dataset
    for path, preprocess_fn in CPT_HF_DATA:
        raw_dataset = load_dataset(path, split="train")
        preprocessed_dataset = raw_dataset.map(preprocess_fn, remove_columns=raw_dataset.column_names)
        dataset = concatenate_datasets([dataset, preprocessed_dataset])

    test_size = len(dataset) % args.batch_size if len(dataset) % args.batch_size != 0 else args.batch_size
    dataset = dataset.train_test_split(test_size=test_size, shuffle=True, seed=args.seed)

    train_dataset = dataset["train"].shuffle()
    val_dataset = dataset["test"].shuffle()

    # training
    target_modules1 = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    target_modules2 = ["embed_tokens", "lm_head"]

    param_group1, param_group2 = [], []
    for name, param in model.named_parameters():
        if any(module_name in name for module_name in target_modules1):
            if "weight" in name:
                param_group1.append(param)
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        elif any(module_name in name for module_name in target_modules2):
            param_group2.append(param)
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
    
    param_groups = [
        {'params': param_group1, 'rank': args.galore_rank, 'update_proj_gap': args.galore_update_proj_gap, 'scale': args.galore_scale, 'proj_type': args.galore_proj_type, 'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': param_group2, 'rank': args.galore_rank, 'update_proj_gap': args.galore_update_proj_gap, 'scale': args.galore_scale, 'proj_type': args.galore_proj_type, 'lr': args.embedding_lr, 'weight_decay': 0.0}
    ]
    optimizer = GaLoreAdamW8bit(param_groups)
    
    total_steps = math.ceil(len(train_dataset) / args.batch_size) * args.epochs
    warmup_steps = int(total_steps * 0.06)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    training_args = UnslothTrainingArguments(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.batch_size,
        num_train_epochs=args.epochs,
        gradient_checkpointing=args.use_gradient_checkpointing,
        max_grad_norm=args.max_grad_norm,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        eval_steps=100,
        save_strategy="epoch",
        seed=args.seed,
        output_dir="./adapters/cpt-galore",
        report_to="wandb"
    )

    trainer_args = {
        "model": model,
        "optimizers": (optimizer, lr_scheduler),
        "tokenizer": tokenizer,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "max_seq_length": args.max_token_length,
        "dataset_text_field": "text",
        "dataset_kwargs": {
            "add_special_tokens": False,
            "append_concat_token": False
        },
        "args": training_args
    }

    trainer = SFTTrainer(**trainer_args)
    
    trainer.train()