
from modules.gpt_modules import gpt_call


class Evaluator:
    def __init__(
            self
            ):
        super().__init__()

        ##############################################################
        # Question & Answer Prompt
        ##############################################################
        self.number_1_qa_prompt = "\n".join([
            "문제: 1. 어떤 특징을 가진 도형을 사다리꼴이라고 약속할 수 있을까요?",
            "답안: 평행한 변이 한 쌍이라도 있는 사각형",
            "지식요소1: 사다리꼴이 사각형임을 알고 있는가?",
            "지식요소2: 평행의 개념을 알고 있는가?",
            "지식요소3: 평행한 변이 한쌍 내지 두 변이라고 표현했는가?",
        ])

        self.number_2_qa_prompt = "\n".join([
        ])

        self.number_3_qa_prompt = "\n".join([
        ])

        self.number_4_qa_prompt = "\n".join([
        ])

        self.number_5_qa_prompt = "\n".join([
        ])

        ##############################################################
        # Criteria_prompt
        ##############################################################
        # Number 1
        self.number_1_1_criteria_text = "\n".join([
            "평가기준: 사다리꼴이 사각형임을 알고 있는가?",
            "- 0점: 사각형이라고 명시하지 않거나 변 4개 또는 꼭짓점 4개라고 설명하지 않는다. 혹은 입체도형의 구성요소인 '면'이라는 용어를 쓴 경우",
            "- 1점: 사각형 또는 변 4개 또는 꼭짓점 4개라고 설명하였을때"
        ])
        self.number_1_1_criteria_text = "\n".join([
            "평가기준: 평행의 개념을 알고 있는가?",
            "- 0점: 평행이라는 용어가 등장하지 않거나 평행을 내포하는 용어를(마주보는 선, 대변, 만나지 않는다 등) 포함하여 설명하지 않는다. 혹은 평행이라는 용어를 썼지만 '평행하지 않는'과 같이 부정어를 포함하여 서술한 경우",
            "- 1점: 평행이라는 용어가 등장하거나, 평행을 내포하는 용어를(마주보는 선, 대변, 만나지 않는다 등) 포함하여 설명한다.",
            "- 2점: 평행이라는 용어를 사용하거나 평행을 내포하는 용어를(마주보는 선, 대변, 만나지 않는다 등) 포함하여 설명 혹은 평행한 변이 한 쌍, 두 변, 두 개의 변을 지칭할 경우"
        ])
        # Number 2
        self.number_2_1_criteria_text = "\n".join([
            "평가기준: 기호를 정확히 찾을 수 있는가?",
            "- 0점: 정답과 일치하지 않을 경우",

        ])


        self.number_4_criteria_text = "\n".join([

        ])
        self.number_5_criteria_text = "\n".join([

        ])


    def eval_number_1(self, df):
        print(df.head())


    def eval_number_2(self, df):
        pass


    def eval_number_3(self, df):
        pass


    def eval_number_4(self, df):
        pass


    def eval_number_5(self, df):
        pass