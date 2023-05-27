
from modules.gpt_modules import gpt_call
from langchain import PromptTemplate

import pandas as pd

from tqdm import tqdm


class Evaluator:
    def __init__(
            self
            ):
        super().__init__()

        ##############################################################
        # Question & Answer Prompt
        ##############################################################
        self.number_1_qa_prompt = "\n".join([
            "문제: ",
            "어떤 특징을 가진 도형을 사다리꼴이라고 약속할 수 있을까요?",
            "답안: ",
            "평행한 변이 한 쌍이라도 있는 사각형",
        ])

        self.number_2_qa_prompt = "\n".join([
            "문제: ",
            "사다리꼴  넓이를  구하는데  필요한  선분  3개를  찾아  기호(예:  선분  ㅈㅊ)를  모두  적고 해당 선분의 이름을 적어보세요. (길이는 모눈종이 칸을 이용하여 구하시오.)",
            "답안: ",
            "선분 ㄱㄹ, 윗변, 6",
            "선분 ㄴㄷ, 아랫변, 14",
            "선분 ㄱㅁ,	높이, 6"
        ])

        self.number_3_qa_prompt = "\n".join([
            "문제: ",
            "아래의 사다리꼴 모양 텃밭을 <예시>와 같이 2개 이상의 도형으로 나누어 텃밭의 넓이를 계산해보세요. 도형에 선분을 그어 표시하고, 어떻게 텃밭의 넓이를 계산했는지 문제 2에서 적은 이름(용어)을 사용하여 식으로 정리해보고, 그림의 과정과 어울리게 구체적으로 과정을 적으세요.",
            "답안: ",
            "정답: 60, 식: (문제 해결 방법에 따라 확인)",
        ])

        self.number_4_qa_prompt = "\n".join([
            "문제: ",
            "아래의 사다리꼴 모양 텃밭에 다른 모양을 덧붙여(추가하여) 텃밭의 넓이를 계산해보세요. 도형에 선분을 그어 표시하고, 어떻게 텃밭의 넓이를 계산했는지 문제 2에서 적은 이름(용어)을 사용하여 식으로 정리해보고, 그림의 과정과 어울리게 구체적으로 과정을 적으세요.",
            "답안: ",
            "(1, 2, 3 중 하나)"
            "1. (14*6)-{(14-6)*^}/2=60",
            "2. (6+14)*6/2=60",
            "3. (14*^)-(5*6/2)-(3*6/2)=60",
        ])

        self.number_5_qa_prompt = "\n".join([
            "문제: ",
            "아래의 사다리꼴 모양 텃밭을 <예시>와 같이 다른 도형으로 바꾸어 텃밭의 넓이를 계산해보세요. 도형에 선분을 그어 표시하고, 어떻게 텃밭의 넓이를 계산했는지 문제 2에서 적은 이름(용어)을 사용하여 식으로 정리해보고, 그림의 과정과 어울리게 구체적으로 과정을 적으세요.",
            "답안: ",
            "(1, 2, 3 중 하나)"
            "1. 아랫변을 연장하고 임의의 꼭짓점과 연결하여 삼각형으로 바꾼 경우",
            "2. 사다리꼴을 높이의 반이 되도록 윗부분을 180도 회전하여 아랫부분과 연결하여 평행사변형으로 바꾼 경우",
            "3. 사다리꼴을 높이의 반이 되도록 수평으로 자르고 윗부분을 수직으로 잘라, 자른 부분 중 왼쪽은 왼쪽으로 180도 회전시켜 아랫부분과 연결하고, 자른 부분 중 오른쪽은 오른쪽으로 180도 회전시켜 아랫부분과 연결하여 직사각형으로 바꾼 경우",
        ])

        ##############################################################
        # Knowledge Components
        ##############################################################
        self.number_1_kc_prompt = "\n".join([
            "지식요소1: ",
            "사다리꼴이 사각형임을 알고 있는가?",
            "지식요소2: ",
            "평행의 개념을 알고 있는가?",
            "지식요소3: ",
            "평행한 변이 한쌍 내지 두 변이라고 표현했는가?",
        ])

        self.number_2_kc_prompt = "\n".join([
            "지식요소1: ",
            "선분 ㄱㄹ, 선분 ㄴㄷ, 선분 ㄱㅁ을 찾을 수 있는가?",
            "지식요소2: ",
            "선분 ㄱㄹ을 윗변으로 적는가?",
            "지식요소3: ",
            "선분 ㄴㄷ을 아랫변으로 적는가?",
            "지식요소4: ",
            "선분 ㄱㅁ을 높이로 적는가?",
            "지식요소5: ",
            "선분 ㄱㄹ의 길이를 올바르게 구하는가?",
            "지식요소6: ",
            "선분 ㄴㄷ의 길이를 올바르게 구하는가?",
            "지식요소7: ",
            "선분 ㄱㅁ의 길이를 올바르게 구하는가?"
        ])

        self.number_3_kc_prompt = "\n".join([
            "지식요소1: ",
            "2개 이상의 도형으로 나누었는가?",
            "지식요소2: ",
            "나눈 도형이 삼각형/사다리꼴/직사각형/평행사변형에 해당하는가?",
            "지식요소3: ",
            "삼각형의 넓이 구하는 방법을 이해하고 계산할 수 있는가?",
            "지식요소4: ",
            "사다리꼴의 넓이를 구하는 방법을 이해하고 계산할 수 있는가?",
            "지식요소5: ",
            "직사각형의 넓이를 구하는 방법을 이해하고 계산할 수 있는가?",
            "지식요소6: ",
            "평행사변형의 넓이를 구하는 방법을 이해하고 계산할 수 있는가?",
            "지식요소7: ",
            "나눈 도형에 적절한 식을 세울 수 있는가?",
            "지식요소8: ",
            "사칙연산을 활용해 정확하게 계산할 수 있는가?",
            "지식요소9: ",
            "식을 세운 방법을 용어를 사용해 설명할 수 있는가?"
        ])

        self.number_4_kc_prompt = "\n".join([
            "지식요소1: ",
            "사다리꼴을 다른 도형으로 바꿀 줄 아는가?",
            "지식요소2: ",
            "덧붙인 도형이 평행사변형 또는 직사각형에 해당하는가?",
            "지식요소3: ",
            "평행사변형으로 바꾸었을 때, 이 평행사변형이 사다리꼴의 2개인 지를 아는가?",
            "지식요소4: ",
            "평행사변형으로 바꾸었을 때, 삼각형의 넓이를 구할 줄 아는가?",
            "지식요소5: ",
            "직사각형으로 바꾸었을 때, 직각삼각형의 넓이를 구할 줄 아는가?",
            "지식요소6: ",
            "사다리꼴의 넓이를 식으로 표현할 수 있는가?",
            "지식요소7: ",
            "사칙연산을 활용해 정확하게 계산할 수 있는가?",
            "지식요소8: ",
            "식을 세운 방법을 용어를 사용해 설명할 수 있는가?"
        ])

        self.number_5_kc_prompt = "\n".join([
            "지식요소1: ",
            "사다리꼴을 다른 도형으로 바꾸었는가?",
            "지식요소2: ",
            "바꾼 도형이 삼각형/평행사변형/직사각형/사다리꼴인가?",
            "지식요소3: ",
            "삼각형으로 바꾸었을 때, 삼각형의 넓이 공식을 알고 있는가?",
            "지식요소4: ",
            "평행사변형으로 바꾸었을 때, 평행사변형의 넓이 공식을 알고 있는가?",
            "지식요소5: ",
            "바꾼 도형에 적절한 식을 세울 수 있는가?",
            "지식요소6: ",
            "정확한 사칙연산으로 올바른 넓이를 구하였는가?",
            "지식요소7: ",
            "식을 세운 방법을 용어를 사용해 설명할 수 있는가?"
        ])


        ##############################################################
        # Criteria Prompt
        ##############################################################
        
        # Number 1
        self.number_1_1_criteria_text = "\n".join([
            "평가기준: ",
            "사다리꼴이 사각형임을 알고 있는가?",
            "- 0점: 사각형이라고 명시하지 않거나 변 4개 또는 꼭짓점 4개라고 설명하지 않는다. 혹은 입체도형의 구성요소인 '면'이라는 용어를 쓴 경우, 혹은 정답을 쓰지 않거나, None이나 NaN으로 표시되어있는 경우",
            "- 1점: 사각형 또는 변 4개 또는 꼭짓점 4개라고 설명하였을때"
        ])
        self.number_1_2_criteria_text = "\n".join([
            "평가기준: ",
            "평행의 개념을 알고 있는가?",
            "- 0점: 평행이라는 용어가 등장하지 않거나 평행을 내포하는 용어를(마주보는 선, 대변, 만나지 않는다 등) 포함하여 설명하지 않는다. 혹은 평행이라는 용어를 썼지만 '평행하지 않는'과 같이 부정어를 포함하여 서술한 경우, 혹은 정답을 쓰지 않거나, None이나 NaN으로 표시되어있는 경우",
            "- 1점: 평행이라는 용어가 등장하거나, 평행을 내포하는 용어를(마주보는 선, 대변, 만나지 않는다 등) 포함하여 설명한다.",
            "- 2점: 평행이라는 용어를 사용하거나 평행을 내포하는 용어를(마주보는 선, 대변, 만나지 않는다 등) 포함하여 설명 혹은 평행한 변이 한 쌍, 두 변, 두 개의 변을 지칭할 경우"
        ])
        
        # Number 2
        self.number_2_1_criteria_text = "\n".join([
            "평가기준: ",
            "기호를 정확히 찾을 수 있는가?",
            "- 0점: 정답과 일치하지 않을 경우, 혹은 정답을 쓰지 않거나, None이나 NaN으로 표시되어있는 경우",
            "- 1점: 정답에서 선분을 서술하지 않았을 경우(ㄴㄷ의 경우 ㄴㄷㅂㄷ도 인정, 기호 순서가 바뀔 경우 인정)",
            "- 2점: 정답과 일치할 경우(변으로 적을 경우 인정, 기호 순서가 바뀔 경우 인정)",
        ])
        self.number_2_2_criteria_text = "\n".join([
            "평가기준: ",
            "각 선분에 대응하는 용어를 정확히 아는가?",
            "- 0점: 정답 중 1개 이하로 일치할 경우, 혹은 정답을 쓰지 않거나, None이나 NaN으로 표시되어있는 경우",
            "- 1점: 정답 중 2개만 일치할 경우",
            "- 2점: 정답과 모두 일치하는 경우(아랫선분과 윗선분은 인정)",
        ])
        self.number_2_3_criteria_text = "\n".join([
            "평가기준: ",
            "각 선분에 대응하는 길이를 올바르게 구하는가?",
            "- 0점: 정답 중 1개 이하로 일치할 경우, 혹은 정답을 쓰지 않거나, None이나 NaN으로 표시되어있는 경우",
            "- 1점: 정답 중 2개만 일치할 경우",
            "- 2점: 정답과 모두 일치하는 경우",
        ])

        # Number 3
        self.number_3_1_criteria_text = "\n".join([
            "평가기준: ",
            "정답과 식을 정확히 서술하여 해결하였는가?",
            "- 0점: 정답과 식이 틀린 경우, 혹은 정답을 쓰지 않거나, None이나 NaN으로 표시되어있는 경우",
            "- 1점: 정답과 식 중에 하나만 맞은 경우",
            "- 2점: 정답과 식을 정확히 서술한 경우",
        ])
        self.number_3_2_criteria_text = "\n".join([
            "평가기준: ",
            "문제해결 방법이 적절한가?",
            "- 0점: 도형을 나누어 해결하지 않은 경우, 혹은 정답을 쓰지 않거나, None이나 NaN으로 표시되어있는 경우",
            "- 1점: 모법답안의 방법과 다르게 나눈 경우",
            "- 2점: 모범답안의 방법과 일치한 경우",
        ])

        # Number 4
        self.number_4_1_criteria_text = "\n".join([
            "평가기준: ",
            "정답과 식을 정확히 서술하여 해결하였는가?",
            "- 0점: 정답과 식이 틀린 경우, 혹은 정답을 쓰지 않거나, None이나 NaN으로 표시되어있는 경우",
            "- 1점: 정답과 식 중에 하나만 맞은 경우",
            "- 2점: 정답과 식을 정확히 서술한 경우",
        ])
        self.number_4_2_criteria_text = "\n".join([
            "평가기준: ",
            "문제해결 방법이 적절한가?",
            "- 0점: 도형을 덧붙여 해결하지 않은 경우, 혹은 정답을 쓰지 않거나, None이나 NaN으로 표시되어있는 경우",
            "- 1점: 모법답안의 방법과 다르게 나눈 경우",
            "- 2점: 모범답안의 방법과 일치한 경우",
        ])

        # Number 5
        self.number_5_1_criteria_text = "\n".join([
            "평가기준: ",
            "정답과 식을 정확히 서술하여 해결하였는가?",
            "- 0점: 정답과 식이 틀린 경우, 혹은 정답을 쓰지 않거나, None이나 NaN으로 표시되어있는 경우",
            "- 1점: 정답과 식 중에 하나만 맞은 경우",
            "- 2점: 정답과 식을 정확히 서술한 경우",
        ])
        self.number_5_2_criteria_text = "\n".join([
            "평가기준: ",
            "문제해결 방법이 적절한가?",
            "- 0점: 다른 도형으로 바꾸어 해결하지 않은 경우, 혹은 정답을 쓰지 않거나, None이나 NaN으로 표시되어있는 경우",
            "- 1점: 모법답안의 방법과 다르게 나눈 경우",
            "- 2점: 모범답안의 방법과 일치한 경우",
        ])

        ##############################################################
        # Persona Prompt
        ##############################################################
        self.persona_prompt = "\n".join([
            "너는 수학 문제를 평가하는 인공지능이다.",
            "문제와 정답, 평가 기준에 따라 학생들의 답안을 평가한다.",
            "평가 결과는 점수와 이유로 나타낸다.",
            "점수는 숫자로 쓴다. 예) 점수: 3",
            "이유는 문장으로 쓴다. 예) 이유: 문제를 잘 읽지 않았다.",
        ])


    ##############################################################
    # Evaluation Number 1_1
    ##############################################################
    def eval_number_1_1(self, df):
        print("eval_number_1_1 start!!")

        user_id_list = []
        user_answer_list = []
        # score와 reason이 포함되어 있음
        bot_response_list = []

        
        for row in tqdm(df.iterrows()):
            try:
                user_id = row[1]['user_id']
                user_answer = row[1]['답안']

                # prompt setting
                number_1_1_prompt_template = "\n".join([
                    self.persona_prompt,
                    self.number_1_qa_prompt, # question and answer prompt
                    self.number_1_kc_prompt, # knowledge component
                    self.number_1_1_criteria_text, # criteria
                    "답안에 대한 점수와 이유를 작성하라.",
                    "답안: {user_answer}",
                    "점수: ",
                    "이유: ",
                ])
                prompt_template = PromptTemplate(
                    input_variables=["user_answer"],
                    template=number_1_1_prompt_template,
                )
                input_prompt = prompt_template.format(user_answer=user_answer)

                bot_response = gpt_call(input_prompt)

                print("bot_response: ", bot_response)

                # append
                user_id_list.append(user_id)
                user_answer_list.append(user_answer)
                bot_response_list.append(bot_response)
            except:
                continue

        # make dataframe
        final_df = pd.DataFrame({
            "user_id": user_id_list,
            "user_answer": user_answer_list,
            "bot_response": bot_response_list,
        })

        return final_df
    
    ##############################################################
    # Evaluation Number 1_2
    ##############################################################
    def eval_number_1_2(self, df):
        print("eval_number_1_2 start!!")

        user_id_list = []
        user_answer_list = []
        # score와 reason이 포함되어 있음
        bot_response_list = []

        for row in tqdm(df.iterrows()):
            try:
                user_id = row[1]['user_id']
                user_answer = row[1]['답안']

                # prompt setting
                number_1_2_prompt_template = "\n".join([
                    self.persona_prompt,
                    self.number_1_qa_prompt, # question and answer prompt
                    self.number_1_kc_prompt, # knowledge component
                    self.number_1_2_criteria_text, # criteria
                    "답안에 대한 점수와 이유를 작성하라.",
                    "답안: {user_answer}",
                    "점수: ",
                    "이유: ",
                ])
                prompt_template = PromptTemplate(
                    input_variables=["user_answer"],
                    template=number_1_2_prompt_template,
                )
                input_prompt = prompt_template.format(user_answer=user_answer)

                bot_response = gpt_call(input_prompt)

                print("bot_response: ", bot_response)

                # append
                user_id_list.append(user_id)
                user_answer_list.append(user_answer)
                bot_response_list.append(bot_response)
            except:
                continue

        # make dataframe
        final_df = pd.DataFrame({
            "user_id": user_id_list,
            "user_answer": user_answer_list,
            "bot_response": bot_response_list,
        })

        return final_df

    ##############################################################
    # Evaluation Number 2_1
    ##############################################################
    def eval_number_2_1(self, df):
        print("eval_number_2_1 start!!")

        user_id_list = []
        user_answer_list = []
        # score와 reason이 포함되어 있음
        bot_response_list = []

        for row in tqdm(df.iterrows()):
            try:
                user_id = row[1]['user_id']
                user_answer = row[1]['답안']

                # prompt setting
                number_2_1_prompt_template = "\n".join([
                    self.persona_prompt,
                    self.number_2_qa_prompt, # question and answer prompt
                    self.number_2_kc_prompt, # knowledge component
                    self.number_2_1_criteria_text, # criteria
                    "답안에 대한 점수와 이유를 작성하라.",
                    "답안: {user_answer}",
                    "점수: ",
                    "이유: ",
                ])
                prompt_template = PromptTemplate(
                    input_variables=["user_answer"],
                    template=number_2_1_prompt_template,
                )
                input_prompt = prompt_template.format(user_answer=user_answer)

                bot_response = gpt_call(input_prompt)

                print("bot_response: ", bot_response)

                # append
                user_id_list.append(user_id)
                user_answer_list.append(user_answer)
                bot_response_list.append(bot_response)
            except:
                continue

        # make dataframe
        final_df = pd.DataFrame({
            "user_id": user_id_list,
            "user_answer": user_answer_list,
            "bot_response": bot_response_list,
        })

        return final_df

    ##############################################################
    # Evaluation Number 2_2
    ##############################################################
    def eval_number_2_2(self, df):
        print("eval_number_2_2 start!!")

        user_id_list = []
        user_answer_list = []
        # score와 reason이 포함되어 있음
        bot_response_list = []

        for row in tqdm(df.iterrows()):
            try:
                user_id = row[1]['user_id']
                user_answer = row[1]['답안']

                # prompt setting
                number_2_2_prompt_template = "\n".join([
                    self.persona_prompt,
                    self.number_2_qa_prompt, # question and answer prompt
                    self.number_2_kc_prompt, # knowledge component
                    self.number_2_2_criteria_text, # criteria
                    "답안에 대한 점수와 이유를 작성하라.",
                    "답안: {user_answer}",
                    "점수: ",
                    "이유: ",
                ])
                prompt_template = PromptTemplate(
                    input_variables=["user_answer"],
                    template=number_2_2_prompt_template,
                )
                input_prompt = prompt_template.format(user_answer=user_answer)

                bot_response = gpt_call(input_prompt)

                print("bot_response: ", bot_response)

                # append
                user_id_list.append(user_id)
                user_answer_list.append(user_answer)
                bot_response_list.append(bot_response)
            except:
                continue

        # make dataframe
        final_df = pd.DataFrame({
            "user_id": user_id_list,
            "user_answer": user_answer_list,
            "bot_response": bot_response_list,
        })

        return final_df
    
    ##############################################################
    # Evaluation Number 2_3
    ##############################################################
    def eval_number_2_3(self, df):
        print("eval_number_2_3 start!!")

        user_id_list = []
        user_answer_list = []
        # score와 reason이 포함되어 있음
        bot_response_list = []

        for row in tqdm(df.iterrows()):
            try:
                user_id = row[1]['user_id']
                user_answer = row[1]['답안']

                # prompt setting
                number_2_3_prompt_template = "\n".join([
                    self.persona_prompt,
                    self.number_2_qa_prompt, # question and answer prompt
                    self.number_2_kc_prompt, # knowledge component
                    self.number_2_3_criteria_text, # criteria
                    "답안에 대한 점수와 이유를 작성하라.",
                    "답안: {user_answer}",
                    "점수: ",
                    "이유: ",
                ])
                prompt_template = PromptTemplate(
                    input_variables=["user_answer"],
                    template=number_2_3_prompt_template,
                )
                input_prompt = prompt_template.format(user_answer=user_answer)

                bot_response = gpt_call(input_prompt)

                print("bot_response: ", bot_response)

                # append
                user_id_list.append(user_id)
                user_answer_list.append(user_answer)
                bot_response_list.append(bot_response)
            except:
                continue

        # make dataframe
        final_df = pd.DataFrame({
            "user_id": user_id_list,
            "user_answer": user_answer_list,
            "bot_response": bot_response_list,
        })

        return final_df


    ##############################################################
    # Evaluation Number 3_1
    ##############################################################
    def eval_number_3_1(self, df):
        print("eval_number_3_1 start!!")

        user_id_list = []
        user_answer_list = []
        user_picture_answer_list = []
        # score와 reason이 포함되어 있음
        bot_response_list = []

        for row in tqdm(df.iterrows()):
            try:
                user_id = row[1]['user_id']
                user_answer = row[1]['답안']
                user_picture_answer = row[1]['텍스트화']

                # prompt setting
                number_3_1_prompt_template = "\n".join([
                    self.persona_prompt,
                    self.number_3_qa_prompt, # question and answer prompt
                    self.number_3_kc_prompt, # knowledge component
                    self.number_3_1_criteria_text, # criteria
                    "답안과 그림답안을 보고 이에 대한 점수와 이유를 작성하라.",
                    "답안: {user_answer}",
                    "그림답안: {user_picture_answer}",
                    "점수: ",
                    "이유: ",
                ])
                prompt_template = PromptTemplate(
                    input_variables=["user_answer", "user_picture_answer"],
                    template=number_3_1_prompt_template,
                )
                input_prompt = prompt_template.format(
                    user_answer=user_answer,
                    user_picture_answer=user_picture_answer,
                    )

                bot_response = gpt_call(input_prompt)

                print("bot_response: ", bot_response)

                # append
                user_id_list.append(user_id)
                user_answer_list.append(user_answer)
                user_picture_answer_list.append(user_picture_answer)
                bot_response_list.append(bot_response)
            except:
                continue

        # make dataframe
        final_df = pd.DataFrame({
            "user_id": user_id_list,
            "user_answer": user_answer_list,
            "user_picture_answer": user_picture_answer_list,
            "bot_response": bot_response_list,
        })

        return final_df

    ##############################################################
    # Evaluation Number 3_2
    ##############################################################
    def eval_number_3_2(self, df):
        print("eval_number_3_2 start!!")

        user_id_list = []
        user_answer_list = []
        user_picture_answer_list = []
        # score와 reason이 포함되어 있음
        bot_response_list = []

        for row in tqdm(df.iterrows()):
            try:
                user_id = row[1]['user_id']
                user_answer = row[1]['답안']
                user_picture_answer = row[1]['텍스트화']

                # prompt setting
                number_3_2_prompt_template = "\n".join([
                    self.persona_prompt,
                    self.number_3_qa_prompt, # question and answer prompt
                    self.number_3_kc_prompt, # knowledge component
                    self.number_3_2_criteria_text, # criteria
                    "답안과 그림답안을 보고 이에 대한 점수와 이유를 작성하라.",
                    "답안: {user_answer}",
                    "그림답안: {user_picture_answer}",
                    "점수: ",
                    "이유: ",
                ])
                prompt_template = PromptTemplate(
                    input_variables=["user_answer", "user_picture_answer"],
                    template=number_3_2_prompt_template,
                )
                input_prompt = prompt_template.format(
                    user_answer=user_answer,
                    user_picture_answer=user_picture_answer,
                    )

                bot_response = gpt_call(input_prompt)

                print("bot_response: ", bot_response)

                # append
                user_id_list.append(user_id)
                user_answer_list.append(user_answer)
                user_picture_answer_list.append(user_picture_answer)
                bot_response_list.append(bot_response)
            except:
                continue

        # make dataframe
        final_df = pd.DataFrame({
            "user_id": user_id_list,
            "user_answer": user_answer_list,
            "user_picture_answer": user_picture_answer_list,
            "bot_response": bot_response_list,
        })

        return final_df


    ##############################################################
    # Evaluation Number 4_1
    ##############################################################
    def eval_number_4_1(self, df):
        print("eval_number_4_1 start!!")

        user_id_list = []
        user_answer_list = []
        user_picture_answer_list = []
        # score와 reason이 포함되어 있음
        bot_response_list = []

        for row in tqdm(df.iterrows()):
            try:
                user_id = row[1]['user_id']
                user_answer = row[1]['답안']
                user_picture_answer = row[1]['텍스트화']

                # prompt setting
                number_4_1_prompt_template = "\n".join([
                    self.persona_prompt,
                    self.number_4_qa_prompt, # question and answer prompt
                    self.number_4_kc_prompt, # knowledge component
                    self.number_4_1_criteria_text, # criteria
                    "답안과 그림답안을 보고 이에 대한 점수와 이유를 작성하라.",
                    "답안: {user_answer}",
                    "그림답안: {user_picture_answer}",
                    "점수: ",
                    "이유: ",
                ])
                prompt_template = PromptTemplate(
                    input_variables=["user_answer", "user_picture_answer"],
                    template=number_4_1_prompt_template,
                )
                input_prompt = prompt_template.format(
                    user_answer=user_answer,
                    user_picture_answer=user_picture_answer,
                    )

                bot_response = gpt_call(input_prompt)

                print("bot_response: ", bot_response)

                # append
                user_id_list.append(user_id)
                user_answer_list.append(user_answer)
                user_picture_answer_list.append(user_picture_answer)
                bot_response_list.append(bot_response)
            except:
                continue

        # make dataframe
        final_df = pd.DataFrame({
            "user_id": user_id_list,
            "user_answer": user_answer_list,
            "user_picture_answer": user_picture_answer_list,
            "bot_response": bot_response_list,
        })

        return final_df

    ##############################################################
    # Evaluation Number 4_2
    ##############################################################
    def eval_number_4_2(self, df):
        print("eval_number_4_2 start!!")

        user_id_list = []
        user_answer_list = []
        user_picture_answer_list = []
        # score와 reason이 포함되어 있음
        bot_response_list = []

        for row in tqdm(df.iterrows()):
            try:
                user_id = row[1]['user_id']
                user_answer = row[1]['답안']
                user_picture_answer = row[1]['텍스트화']

                # prompt setting
                number_4_2_prompt_template = "\n".join([
                    self.persona_prompt,
                    self.number_4_qa_prompt, # question and answer prompt
                    self.number_4_kc_prompt, # knowledge component
                    self.number_4_2_criteria_text, # criteria
                    "답안과 그림답안을 보고 이에 대한 점수와 이유를 작성하라.",
                    "답안: {user_answer}",
                    "그림답안: {user_picture_answer}",
                    "점수: ",
                    "이유: ",
                ])
                prompt_template = PromptTemplate(
                    input_variables=["user_answer", "user_picture_answer"],
                    template=number_4_2_prompt_template,
                )
                input_prompt = prompt_template.format(
                    user_answer=user_answer,
                    user_picture_answer=user_picture_answer,
                    )

                bot_response = gpt_call(input_prompt)

                print("bot_response: ", bot_response)

                # append
                user_id_list.append(user_id)
                user_answer_list.append(user_answer)
                user_picture_answer_list.append(user_picture_answer)
                bot_response_list.append(bot_response)
            except:
                continue

        # make dataframe
        final_df = pd.DataFrame({
            "user_id": user_id_list,
            "user_answer": user_answer_list,
            "user_picture_answer": user_picture_answer_list,
            "bot_response": bot_response_list,
        })

        return final_df


    ##############################################################
    # Evaluation Number 5_1
    ##############################################################
    def eval_number_5_1(self, df):
        print("eval_number_5_1 start!!")

        user_id_list = []
        user_answer_list = []
        user_picture_answer_list = []
        # score와 reason이 포함되어 있음
        bot_response_list = []

        for row in tqdm(df.iterrows()):
            try:
                user_id = row[1]['user_id']
                user_answer = row[1]['답안']
                user_picture_answer = row[1]['텍스트화']

                # prompt setting
                number_5_1_prompt_template = "\n".join([
                    self.persona_prompt,
                    self.number_5_qa_prompt, # question and answer prompt
                    self.number_5_kc_prompt, # knowledge component
                    self.number_5_1_criteria_text, # criteria
                    "답안과 그림답안을 보고 이에 대한 점수와 이유를 작성하라.",
                    "답안: {user_answer}",
                    "그림답안: {user_picture_answer}",
                    "점수: ",
                    "이유: ",
                ])
                prompt_template = PromptTemplate(
                    input_variables=["user_answer", "user_picture_answer"],
                    template=number_5_1_prompt_template,
                )
                input_prompt = prompt_template.format(
                    user_answer=user_answer,
                    user_picture_answer=user_picture_answer,
                    )

                bot_response = gpt_call(input_prompt)

                print("bot_response: ", bot_response)

                # append
                user_id_list.append(user_id)
                user_answer_list.append(user_answer)
                user_picture_answer_list.append(user_picture_answer)
                bot_response_list.append(bot_response)
            except:
                continue

        # make dataframe
        final_df = pd.DataFrame({
            "user_id": user_id_list,
            "user_answer": user_answer_list,
            "user_picture_answer": user_picture_answer_list,
            "bot_response": bot_response_list,
        })

        return final_df
    
    ##############################################################
    # Evaluation Number 5_2
    ##############################################################
    def eval_number_5_2(self, df):
        print("eval_number_5_2 start!!")

        user_id_list = []
        user_answer_list = []
        user_picture_answer_list = []
        # score와 reason이 포함되어 있음
        bot_response_list = []

        for row in tqdm(df.iterrows()):
            try:
                user_id = row[1]['user_id']
                user_answer = row[1]['답안']
                user_picture_answer = row[1]['텍스트화']

                # prompt setting
                number_5_2_prompt_template = "\n".join([
                    self.persona_prompt,
                    self.number_5_qa_prompt, # question and answer prompt
                    self.number_5_kc_prompt, # knowledge component
                    self.number_5_2_criteria_text, # criteria
                    "답안과 그림답안을 보고 이에 대한 점수와 이유를 작성하라.",
                    "답안: {user_answer}",
                    "그림답안: {user_picture_answer}",
                    "점수: ",
                    "이유: ",
                ])
                prompt_template = PromptTemplate(
                    input_variables=["user_answer", "user_picture_answer"],
                    template=number_5_2_prompt_template,
                )
                input_prompt = prompt_template.format(
                    user_answer=user_answer,
                    user_picture_answer=user_picture_answer
                    )

                bot_response = gpt_call(input_prompt)

                print("bot_response: ", bot_response)

                # append
                user_id_list.append(user_id)
                user_answer_list.append(user_answer)
                user_picture_answer_list.append(user_picture_answer)
                bot_response_list.append(bot_response)
            except:
                continue

        # make dataframe
        final_df = pd.DataFrame({
            "user_id": user_id_list,
            "user_answer": user_answer_list,
            "user_picture_answer": user_picture_answer_list,
            "bot_response": bot_response_list,
        })

        return final_df