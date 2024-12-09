import torch
import torch.nn.functional as F
from trl import SFTTrainer
from transformers import AutoTokenizer


TOKENIZER = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-7B-Instruct")


CPTTemplate = """
{text}<|endoftext|>
""".strip()


InstQATempalateV1 = """
<|im_start|>system
{instruction}<|im_end|>
{question}
{answer}<|endoftext|>
""".strip()


InstQATempalateV2 = """
<|im_start|>system
{instruction}<|im_end|>
### 질문: {question}
### 정답:
{answer}<|endoftext|>
""".strip()


InstMCQATemplate = """
<|im_start|>system
{instruction}<|im_end|>
### 질문: {question}
### 선택지:
{option}
### 정답:
{answer}<|endoftext|>
""".strip()


BASE_INST = """
1. 답변은 기존 사전 학습된 형식(존댓말, 구체적 설명 등)을 유지하세요.
2. 질문이나 요청에는 예시를 들어 구체적이고 상세히 답변하세요.
""".strip()


UNLEARNING_INST = """
1. 답변은 기존 사전 학습된 형식(존댓말, 구체적 설명 등)을 유지하세요.
2. 질문이나 요청에는 예시를 들어 구체적이고 상세히 답변하세요.
3. 이후 제공되는 정보를 기반으로 기존 관련된 내용을 잊고 새로운 정보를 학습하세요.
""".strip()


def formatCPTV1(example):
    text = example.pop("text", None)
    if text is not None:
        return {"text": CPTTemplate.format(text=text)}

    for key in ["Question", "question", "prompt"]:
        question = example.pop(key, None)
        if question is not None:
            break
    for key in ["Answer", "answer", "response"]:
        answer = example.pop(key, None)
        if answer is not None:
            break
    return {"text": CPTTemplate.format(text=f"{question}\n{answer}")}


def formatCPTV2(example):
    question = eval(example["Question"])["Question"]
    answer = {example['Answer']}
    return {"text": CPTTemplate.format(text=f"{question}\n{answer}")}


def formatCPTV3(example):
    if " | Additional_info: " in example["question"]:
        question, info = example["question"].split(" | Additional_info: ")
    else:
        question, info = example["question"], None
    question = question.lstrip(question.split()[0])
    if info is not None:
        question += f"\n{info}"

    option_dict = eval(example["options"])
    explanation = str(example["rationale"]).replace('\\n', '\n')
    correct = example["correct_option"]
    
    option_mapping = {"opt_1": "A", "opt_2": "B", "opt_3": "C", "opt_4": "D"}
    correct = option_mapping[correct]
    for k, v in option_mapping.items(): 
        explanation = explanation.replace(k ,v)

    option = "\n".join([f"{k}. {v}"for k, v in option_dict.items()])
    explanation = explanation.split("### ")[1].lstrip("설명:").strip()
    answer = f"{correct}. {option_dict[correct]}\n\n{explanation}"

    return {"text": CPTTemplate.format(text=f"{question}\n{option}\n\n{answer}")}


def formatITV1(example):
    instruction = UNLEARNING_INST if example["use_machine_unlearning"] else BASE_INST
    for key in ["Question", "question", "prompt"]:
        question = example.pop(key, None)
        if question is not None:
            break
    for key in ["Answer", "answer", "response"]:
        answer = example.pop(key, None)
        if answer is not None:
            break
    inputs = TOKENIZER(InstQATempalateV1.format(instruction=instruction, question=question, answer=answer), add_special_tokens=False)

    total_len = len(inputs["input_ids"])
    answer_len = len(TOKENIZER(f"{answer}<|endoftext|>", add_special_tokens=False)["input_ids"])
    inputs["labels"] = [-100] * (total_len-answer_len) + inputs["input_ids"][-answer_len:]
    return inputs


def formatITV2(example):
    instruction = UNLEARNING_INST if example["use_machine_unlearning"] else BASE_INST
    for key in ["Question", "question", "prompt"]:
        question = example.pop(key, None)
        if question is not None:
            break
    for key in ["Answer", "answer", "response"]:
        answer = example.pop(key, None)
        if answer is not None:
            break
    inputs = TOKENIZER(InstQATempalateV1.format(instruction=instruction, question=question, answer=answer), add_special_tokens=False)

    total_len = len(inputs["input_ids"])
    answer_len = len(TOKENIZER(f"{answer}<|endoftext|>", add_special_tokens=False)["input_ids"])
    inputs["labels"] = [-100] * (total_len-answer_len) + inputs["input_ids"][-answer_len:]
    return inputs


def formatITV3(example):
    instruction = UNLEARNING_INST if example["use_machine_unlearning"] else BASE_INST
    if " | Additional_info: " in example["question"]:
        question, info = example["question"].split(" | Additional_info: ")
    else:
        question, info = example["question"], None
    question = question.lstrip(question.split()[0])
    if info is not None:
        question += f"\n{info}"

    option_dict = eval(example["options"])
    explanation = str(example["rationale"]).replace('\\n', '\n')
    correct = example["correct_option"]
    
    option_mapping = {"opt_1": "A", "opt_2": "B", "opt_3": "C", "opt_4": "D"}
    correct = option_mapping[correct]
    for k, v in option_mapping.items(): 
        explanation = explanation.replace(k ,v)

    option = "\n".join([f"{k}. {v}"for k, v in option_dict.items()])
    explanation = explanation.split("### ")[1].lstrip("설명:").strip()
    answer = f"{correct}. {option_dict[correct]}\n\n{explanation}"
    inputs = TOKENIZER(InstQATempalateV1.format(instruction=instruction, question=question, answer=answer), add_special_tokens=False)

    total_len = len(inputs["input_ids"])
    answer_len = len(TOKENIZER(f"{answer}<|endoftext|>", add_special_tokens=False)["input_ids"])
    inputs["labels"] = [-100] * (total_len-answer_len) + inputs["input_ids"][-answer_len:]
    return inputs


class CustomTrainer(SFTTrainer):
    def __init__(self, **kwargs):
        self.base_model = kwargs.pop("base_model")
        self.kl_coef = kwargs.pop("kl_coef")
        super().__init__(**kwargs)
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)

        with torch.no_grad():
            base_outputs = self.base_model(**inputs).logits
            base_dist = F.softmax(base_outputs, dim=-1)
        
        lora_outputs = outputs.logits
        lora_dist = F.log_softmax(lora_outputs, dim=-1)
        mask = (inputs["labels"] != -100).float().unsqueeze(-1)

        kl_loss = (F.kl_div(lora_dist, base_dist, reduction="none") * mask).sum() / mask.sum()
        loss += self.kl_coef * kl_loss
        return (loss, outputs) if return_outputs else loss

