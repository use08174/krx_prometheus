import json

from .utils import to_GPT
from .utils import matches_structure

structure_example = [{
    "Rationale": "<detailed reasoning and logical backups for solution>",
    "Answer": "<Selected answer>"
}]

system_prompt = f"""You are an financial expert in Korea.
From given question, you must provide rationale and answer in order to create helpful answer sheet for students.

Strictly follow the given structure: {structure_example}
** Return only pure JSON format without any code block or delimiters. **
** Make sure that the response does not create JSON decode error. **
"""

def SOL_GPT(content, system=system_prompt):
    for attempt in range(3):
        try:
            response = to_GPT(system, content)
            response_content = response['choices'][0]['message']['content']
            # print(f"Response Attempt {attempt+1}: {response_content}")

            response_data = json.loads(response_content)
            expected_structure = structure_example
            
            if matches_structure(response_data, expected_structure):
                return response_data
            
            else:
                print(f'출력 구조가 맞지 않습니다. 다시 시도합니다... {attempt+1}')
                content += '\n Please strictly follow the given JSON structure in your response.'

        except (json.JSONDecodeError, SyntaxError, ValueError) as e:
            last_error = str(e)

    return {"error": f"Failed to generate data in the expected structure after 3 attempts. Last error: {last_error}"}