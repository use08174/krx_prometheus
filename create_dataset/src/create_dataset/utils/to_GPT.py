import openai

# OpenAI API 클라이언트 초기화
client = openai.OpenAI(api_key="sk-j9K4kTjcqS_z7shAfRDykgfkqOZDeOsSNlHTuRHl9CT3BlbkFJ1EDZPZcjhoEZFr6S3n97cOmVJ8FMzNviUdlf3R0jEA")

# GPT 호출 함수
def to_GPT(system, prompt, model_type = "gpt-4o-mini"):
    """
    model_type에 gpt-4o-mini, 혹은 gpt-4o-2024-11-20
    """
    response = client.chat.completions.create(
        model=model_type,
        messages=[
            {"role": "system", "content": f"{system}"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=3000,
        temperature=0.7
    ).to_dict()
    return response