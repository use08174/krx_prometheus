import openai
import re

# OpenAI API 클라이언트 초기화
client = openai.OpenAI(api_key="sk-j9K4kTjcqS_z7shAfRDykgfkqOZDeOsSNlHTuRHl9CT3BlbkFJ1EDZPZcjhoEZFr6S3n97cOmVJ8FMzNviUdlf3R0jEA")

# type1) GPT4o-mini로 채점하기
structure_prompt = '{score}'
rule_prompt = '''**Follow the specified structure strictly when providing responses.**
Assign a score between 0.00 and 1.00, rounded to two decimal places. A score of 1.00 indicates an optimal response, and 0.00 indicates a poor one.
You must only generate numbers.
Avoid giving exact scores of 0.00 or 1.00 unless necessary.'''

prompt = structure_prompt + rule_prompt
def benchmark_by_gpt(input, Q_type, prompt=prompt):
    max_attempts = 3  # 최대 재시도 횟수
    attempts = 0

    while attempts < max_attempts:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a seasoned financial analyst. Your role is to assess the performance of financial LLMs based on their scores."},
                {"role": "user", "content": f"Prompt: {prompt} \n Model's Response: {input} \n Q_type: {Q_type}"}
            ],
            max_tokens=3000,
            temperature=0.7
        ).to_dict()

        score_text = response['choices'][0]['message']['content'].strip()
        # print(score_text)

        # 점수 형식이 맞는지 검증
        if re.fullmatch(r"0\.\d{2}|1\.00", score_text):
            return_value = f'Q_type: {Q_type} | Score: {score_text}'
            return return_value

        attempts += 1

    # 3번 시도하고 안되면 걍 포기
    return f'Q_type: {Q_type} | Score: Invalid score after {max_attempts} attempts'