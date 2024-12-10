import json

from .utils import to_GPT
from .utils import matches_structure

role_dict = {
    'AMA': '''Your job is creating multi-hop reasoning questions in fluent Korean. You will be given a part of a text. Make a question based on it. The question should require multiple steps of reasoning related to the text. Return the question only without any other text.

**Step-by-Step Instructions**:
1. Review the provided content to identify essential financial and accounting principles.
2. Generate **detailed, multi-paragraph interpretations** of each concept, framing them as questions and answers that provide unique, well-rounded perspectives.
3. Develop **multiple, rich examples and scenarios** to illustrate each interpretation, incorporating step-by-step explanations and optional variations.
4. Avoid redundancy by ensuring each question and answer explores fresh angles, offering **expansive insights** and practical applications.
5. Verify that all data remains conceptually broad and avoids specific company or regional details.

**Goal**: Create a vast, high-quality, deeply informative questions that enrich foundational financial knowledge and are optimized for LLM fine-tuning. Each answer should resemble a well-researched article or lesson in its depth and coverage.''',

    'PRO': '''Your job is to ensure the precision, clarity, and depth of questions for LLM fine-tuning in financial and accounting topics. Your role is to review each questions critically and provide elaborate, multi-paragraph feedback to ensure exceptional quality. Your responses should offer detailed critiques, including both strengths and weaknesses, and propose comprehensive improvements.

**Step-by-Step Instructions**:
1. Carefully review each questions from AMA_GPT for accuracy, logical structure, and relevance, offering **in-depth critiques that span multiple perspectives.**
2. Elaborate on terminology, provide definitions, and recommend clear restructuring or alternative wording where necessary.
3. Provide detailed justifications for all suggested changes, using specific examples to illustrate points and improve clarity.
4. Remove redundancies while enriching the dataset with precise, actionable feedback and suggestions for enhancement.
5. Ensure that all data is polished and of the highest quality, with feedback that reads like a comprehensive review or audit.

**Goal**: Deliver a refined, detailed, and highly polished questions that reflects universal financial and accounting principles, ready for LLM fine-tuning. Each critique should be lengthy, nuanced, and constructive, offering deep insights into the data’s improvement.''',

    'HAM': '''You are a skilled financial expert in Korea. Make a response for the question. DO NOT introduce yourself. 
    You must harmonize and finalize questions, integrating AMA_GPT’s broad ideas and PRO_GPT’s detailed feedback to produce a high-quality, coherent question. 
    After that, create a detailed step by step answer for high quality question.
    Your role is to create a polished, elaborative Q&A set that includes nuanced explanations, balanced perspectives, and universally applicable financial principles.

**Step-by-Step Instructions**:
1. Combine AMA_GPT’s expansive ideas with PRO_GPT’s detailed critiques to create **multi-paragraph, fully-developed questions** that reflect the best of both inputs.
2. Expand on each concept with **extensive, well-rounded explanations,** using examples, step-by-step walkthroughs, and context-specific scenarios to enhance understanding.
3. Ensure clarity and logical flow while balancing depth and accessibility, ensuring each quesitons could serve as a standalone educational resource.
4. Remove redundancies, refine phrasing, and confirm that each entry offers new insights, covering key financial and accounting concepts comprehensively.
5. Finalize Q&A pairs with an emphasis on detail, ensuring they are structured as fully-formed lessons or case studies ready for LLM fine-tuning.

**Goal**: Create a polished, rich, and deeply informative dataset optimized for LLM fine-tuning, with each Q&A pair offering a long, detailed exploration of financial principles that is both engaging and educational.''',
}

personality_dict = {
    'AMA': '''Imaginative, experimental, and highly detailed. You aim to supply an extensive range of data with richly detailed examples and multi-dimensional perspectives. You embrace depth and length, exploring uncharted details with enthusiasm.''',
    'PRO': '''Conservative, meticulous, and deeply analytical. You aim for rigorous reviews with exhaustive feedback that identifies risks, inconsistencies, and areas for enhancement. You prioritize thorough explanations and actionable recommendations.''',
    'HAM': '''Collaborative, balanced, and analytically thorough. You aim to integrate feedback into rich, multi-faceted data entries that balance depth and coherence. You ensure every Q&A pair is educational and highly detailed, with nuanced insights.''',
}

attitude_dict = {
    'AMA': '''You approach data generation with enthusiasm and creativity, aiming to expand every idea into a detailed, multi-faceted exploration.''',
    'PRO': '''You approach data review with meticulous care, providing detailed suggestions, corrections, and expansions to ensure maximum depth and precision.''',
    'HAM': '''You enhance data by harmonizing exploratory and analytical approaches, striving for maximum detail and completeness in every entry.''',
}


structure_dict = {
    'AMA': [{"Question": "<Sample question>"}],
    'PRO': [{"Critique": "<Sample critique>"}],
    'HAM': [{"Question": "<Sample question>", "Answer": "<Sample answer>", "Final_Review": "<Sample final review>"}],
}

def AMA_GPT(chunk):
    """
    주어진 정보(chunk) 기반 데이터 생성
    """
    system = f'''
    ### Role: {role_dict['AMA']}\n
    ### Personality: {personality_dict['AMA']}\n
    ### Attitude: {attitude_dict['AMA']}\n
    ** All output must be Korean **
    '''

    prompt = f'''
    ### Information: {chunk}

    ** You must strictly follow given structure. ** \n
    ** Return only pure JSON format without any code block or delimiters. **\n
    Structure example: {json.dumps(structure_dict['AMA'])}
    '''
    
    for attempt in range(3):  # 최대 3회 시도
        try:
            # GPT 모델에 프롬프트 전달하여 응답 받기
            response = to_GPT(system, prompt)
            response_content = response['choices'][0]['message']['content']
            # print(f"Response Attempt {attempt+1}: {response_content}")  # 응답 내용 출력

            # 응답 내용을 JSON 형식으로 로드
            response_data = json.loads(response_content)  # JSON 구문으로 안전하게 로드
            expected_structure = structure_dict['AMA']
            
            # 구조가 일치하는지 확인
            if matches_structure(response_data, expected_structure):
                return response_data  # 올바른 구조의 응답 반환
            
            else:
                print(f'출력 구조가 맞지 않습니다. 다시 시도합니다... {attempt+1}')
                prompt += '\n Please strictly follow the given JSON structure in your response.'

        except (json.JSONDecodeError, SyntaxError, ValueError) as e:
            last_error = str(e)
            print(f"에러 발생: {last_error}")

    return {"error": f"Failed to generate data in the expected structure after 3 attempts. Last error: {last_error}"}

def PRO_GPT(AMA_GPT_generated_QA, given_chunk):
    system = f'''
    ### Role: {role_dict['PRO']}\n
    ### Personality: {personality_dict['PRO']}\n
    ### Attitude: {attitude_dict['PRO']}\n
    ** All output must be Korean **
    '''

    prompt = f'''
    You must create detailed critique by analyzing the given generation from AMA_GPT.\n
    ### AMA_GPT's generation: {AMA_GPT_generated_QA}
    ### Source Information: {given_chunk}

    ** You must strictly follow the given JSON structure. ** \n
    ** Return only pure JSON format without any code block or delimiters. **\n
    Structure example: {json.dumps(structure_dict['PRO'])}
    '''
    response = to_GPT(system, prompt, model_type='gpt-4o-2024-11-20')
    response_content = response['choices'][0]['message']['content']
    # print(f"Response Attempt {attempt+1}: {response_content}")

    return response_content
    
    # for attempt in range(3):
    #     try:
    #         response = to_GPT(system, prompt, model_type='gpt-4o-2024-11-20')
    #         response_content = response['choices'][0]['message']['content']
    #         # print(f"Response Attempt {attempt+1}: {response_content}")

    #         response_data = json.loads(response_content)
    #         expected_structure = structure_dict['PRO']
            
    #         if matches_structure(response_data, expected_structure):
    #             return response_data
            
    #         else:
    #             print(f'출력 구조가 맞지 않습니다. 다시 시도합니다... {attempt+1}')
    #             prompt += '\n Please strictly follow the given JSON structure in your response.'

    #     except (json.JSONDecodeError, SyntaxError, ValueError) as e:
    #         last_error = str(e)

    # return {"error": f"Failed to generate data in the expected structure after 3 attempts. Last error: {last_error}"}

def HAM_GPT(given_chunk, AMA_GPT_generated_QA, PRO_GPT_generated_critique):
    system = f'''
    ### Role: {role_dict['HAM']}\n
    ### Personality: {personality_dict['HAM']}\n
    ### Attitude: {attitude_dict['HAM']}\n
    ** All output must be Korean **
    '''

    prompt = f'''
    Perform a final review of all data points, ensuring they align with both creative breadth and logical precision.\n
    ### AMA_GPT's generation: {AMA_GPT_generated_QA}
    ### PRO_GPT's critique: {PRO_GPT_generated_critique}
    ### Source Information: {given_chunk}

    ** You must strictly follow the given JSON structure. ** \n
    ** Return only pure JSON format without any code block or delimiters. **\n
    Structure example: {json.dumps(structure_dict['HAM'])}
    '''
    
    for attempt in range(3):
        try:
            response = to_GPT(system, prompt, model_type='gpt-4o-2024-11-20')
            response_content = response['choices'][0]['message']['content']
            # print(f"Response Attempt {attempt+1}: {response_content}")

            response_data = json.loads(response_content)
            expected_structure = structure_dict['HAM']
            
            if matches_structure(response_data, expected_structure):
                return response_data
            
            else:
                print(f'출력 구조가 맞지 않습니다. 다시 시도합니다... {attempt+1}')
                prompt += '\n Please strictly follow the given JSON structure in your response.'

        except (json.JSONDecodeError, SyntaxError, ValueError) as e:
            last_error = str(e)

    return {"error": f"Failed to generate data in the expected structure after 3 attempts. Last error: {last_error}"}