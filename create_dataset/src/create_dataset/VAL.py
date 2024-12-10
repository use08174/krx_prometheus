import json

from .utils import to_GPT
from .utils import matches_structure

system_prompt = f"""주어진 두 풀이가 동일한 내용일 경우 True, 그렇지 않을 경우 False를 반환하세요.

** 반드시 True, 혹은 False 중 하나만 반환하세요. **
"""
valid_list = [True, 'True', False, 'False']

def VAL_GPT(content, system=system_prompt):
    for attempt in range(3):
        try:
            response = to_GPT(system, content)
            response_content = response['choices'][0]['message']['content']
        
            if response_content in valid_list:
                return response_content
            
            else:
                print(f'출력 구조가 맞지 않습니다. 다시 시도합니다... {attempt+1}')
                content += '\n Please strictly follow the given JSON structure in your response.'

        except (json.JSONDecodeError, SyntaxError, ValueError) as e:
            last_error = str(e)

    return {"error": f"Failed to generate data in the expected structure after 3 attempts. Last error: {last_error}"}