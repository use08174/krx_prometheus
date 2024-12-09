import os
import random
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from trl import SFTTrainer
from datasets  import Dataset, load_dataset, concatenate_datasets
from unsloth import FastLanguageModel, is_bfloat16_supported, UnslothTrainingArguments

from train_utils import CustomTrainer, formatCPTV1, formatCPTV2, formatCPTV3


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

    # lora setting
    parser.add_argument("--lora_r", default=128, type=int)
    parser.add_argument("--lora_alpha", default=32, type=int)
    parser.add_argument("--lora_dropout", default=0.0, type=float)
    parser.add_argument("--use_rslora", action="store_true")

    # training arguments setting
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--embedding_lr", default=1e-5, type=float)
    parser.add_argument("--lr_scheduler", default="linear", type=str)
    parser.add_argument("--lr_warmup_ratio", default=0.06, type=float)
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument("--max_grad_norm", default=2.0, type=float)
    parser.add_argument("--use_gradient_checkpointing", action='store_true')

    # knowledge distillation setting
    parser.add_argument("--use_knowledge_distillation", action='store_true')
    parser.add_argument("--kl_coef", default=0.2, type=float)
    
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
        load_in_4bit=True,
        max_seq_length=args.max_token_length,
        device_map="auto"
    )
    model.config.use_cache = args.use_cache
    tokenizer.padding_side = "left"

    # create PEFT model with LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        random_state=args.seed,
        use_rslora=args.use_rslora, 
        loftq_config=None
    )

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
    training_args = UnslothTrainingArguments(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.batch_size,
        num_train_epochs=args.epochs,
        gradient_checkpointing=args.use_gradient_checkpointing,
        max_grad_norm=args.max_grad_norm,
        learning_rate=args.lr,
        embedding_learning_rate=args.embedding_lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.lr_warmup_ratio,
        weight_decay=args.weight_decay,
        optim="adamw_8bit",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        eval_steps=100,
        save_strategy="epoch",
        seed=args.seed,
        output_dir="./adapters/cpt-lora",
        report_to="wandb"
    )

    trainer_args = {
        "model": model,
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

    if args.use_knowledge_distillation:
        base_model, _ = FastLanguageModel.from_pretrained(
            model_name=args.model,
            dtype=None,
            load_in_4bit=True,
            max_seq_length=args.max_token_length,
            device_map="auto"
        )
        FastLanguageModel.for_inference(base_model)
        trainer_args["base_model"] = base_model
        trainer_args["kl_coef"] = args.kl_coef

        trainer = CustomTrainer(**trainer_args)
    else:
        trainer = SFTTrainer(**trainer_args)
    
    trainer.train()