# KRX 금융 언어 모델 경진대회

## Dataset

![generate_dataset](./images/generate_dataset.png)

## Training

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

## Team Member

| 이름 | 김도현 | 문재원 | 심수민 | 정연석 |
| :-----------: | :----------------------------------------------: | :----------------------------------: | :----------------------------------------: |:----------------------------------------: |
| Github ID | [@Dohyeon-Kim1](https://github.com/Dohyeon-Kim1) | [@lumiere-on](https://github.com/lumiere-on) | [@use08174](https://github.com/use08174) | [@sit-in-a-row](https://github.com/sit-in-a-row)