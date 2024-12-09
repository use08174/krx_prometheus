# KRX 금융 언어 모델 경진대회

### Dataset

### Continued Pretraining (CPT)

```bash
# LoRA
python continued_pretraining_lora.py \
--model=unsloth/Qwen2.5-7B-Instruct \
--use_gradient_checkpointing

# LoRA + Knowledge Distillation
python continued_pretraining_lora.py \
--model=unsloth/Qwen2.5-7B-Instruct \
--use_gradient_checkpointing \
--use_knowledge_distillation

# GaLore
python continued_pretraining_galore.py \
--model=unsloth/Qwen2.5-7B-Instruct \
--use_gradient_checkpointing
```

### Instruction Tuning (IT)
```bash
# LoRA
python instruction_tuning_lora.py \
--model={continued_pretrained_model_path} \
--use_gradient_checkpointing

# LoRA + Knowledge Distillation
python instruction_tuning_lora.py \
--model={continued_pretrained_model_path} \
--use_gradient_checkpointing \
--use_knowledge_distillation
```

### Team Member