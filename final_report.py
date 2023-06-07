import pandas as pd
import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


"""
202011350 인공지능학과 송성진
"""

class SimpleChatBot:
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath, nrows = 500)
        self.questions = self.data['Q'].tolist()  # 질문열만 뽑아 파이썬 리스트로 저장
        self.answers = self.data['A'].tolist()   # 답변열만 뽑아 파이썬 리스트로 저장
        self.vectorizer = TfidfVectorizer()
        self.question_vectors = self.vectorizer.fit_transform(self.questions)  # 질문을 TF-IDF로 변환


    def find_min_dist(self)->int:
        """
        레벤슈타인 거리가 최소가 되는 인덱스 찾고 반환하는 함수
        """
        min_dist_idx = self.data['distance'].argmin()
        return min_dist_idx


    def find_best_answer(self, index_min_dist:int)->str:
        """
        질문과 가장 유사한 답의 인덱스에 해당하는 답변을 반환하는 함수
        """
        answer = self.data.loc[index_min_dist, 'A']
        return answer


# 레벤슈타인 거리 구하기
def calc_distance(a, b):
    """
    param : a, b는 각각 레벤슈타인 거리를 구하고자 하는 문장 2개
    """
    ''' 레벤슈타인 거리 계산하기 '''
    if a == b: return 0 # 같으면 0을 반환
    a_len = len(a) # a 길이
    b_len = len(b) # b 길이
    if a == "": return b_len
    if b == "": return a_len
    # 2차원 표 (a_len+1, b_len+1) 준비하기 --- (※1)
    # matrix 초기화의 예 : [[0, 1, 2, 3], [1, 0, 0, 0, 0], [2, 0, 0, 0, 0], [3, 0, 0, 0, 0], [4, 0, 0, 0, 0]]
    # [0, 1, 2, 3]
    # [1, 0, 0, 0]
    # [2, 0, 0, 0]
    # [3, 0, 0, 0] 
    matrix = [[] for i in range(a_len+1)] # 리스트 컴프리헨션을 사용하여 1차원 초기화
    for i in range(a_len+1): # 0으로 초기화
        matrix[i] = [0 for j in range(b_len+1)]  # 리스트 컴프리헨션을 사용하여 2차원 초기화
    # 0일 때 초깃값을 설정
    for i in range(a_len+1):
        matrix[i][0] = i
    for j in range(b_len+1):
        matrix[0][j] = j
    # 표 채우기 --- (※2)
    # print(matrix,'----------')
    for i in range(1, a_len+1):
        ac = a[i-1]
        # print(ac,'=============')
        for j in range(1, b_len+1):
            bc = b[j-1] 
            # print(bc)
            cost = 0 if (ac == bc) else 1  #  파이썬 조건 표현식 예:) result = value1 if condition else value2
            matrix[i][j] = min([
                matrix[i-1][j] + 1,     # 문자 제거: 위쪽에서 +1
                matrix[i][j-1] + 1,     # 문자 삽입: 왼쪽 수에서 +1   
                matrix[i-1][j-1] + cost # 문자 변경: 대각선에서 +1, 문자가 동일하면 대각선 숫자 복사
            ])
            # print(matrix)
        # print(matrix,'----------끝')
    return matrix[a_len][b_len]


# CSV 파일 경로를 지정하세요.
filepath = 'C:/Users/Administrator/Desktop/python_workspace/chatbot/ChatbotData.csv'

# 간단한 챗봇 인스턴스를 생성합니다.
chatbot = SimpleChatBot(filepath)

# '종료'라는 단어가 입력될 때까지 챗봇과의 대화를 반복합니다.
while True:
    input_sentence = input('You: ')
    if input_sentence.lower() == '종료':
        break
    """
    1. '학습데이터의 질문'과 'chat의 질문'의 유사도를 레벤슈타인 거리를 이용해 구하기
    학습데이터의 질문 : chatbot.data['Q'], (11823,) 벡터
    chat의 질문 : input sentence from user
    """
    # ChatbotData.csv의 'Q' column에 대해 일괄적으로 레벤슈타인 거리 구하는 공식을 적용, 인자로 input_sentence를 넘겨 줌.
    chatbot.data['distance'] = chatbot.data['Q'].apply(
        calc_distance,
        args=(input_sentence,)
    )
    
    """
    2. 'chat 질문'과 '레벤슈타인 거리'와 가장 유사한 학습데이터의 질문의 인덱스를 구하기
    가장 유사한 학습데이터란 distance(레벤슈타인 거리)가 가장 낮은 값을 뜻함.
    """
    # distance(레벤슈타인 거리)가 최소가 되는 인덱스 반환
    min_dist_idx = chatbot.data['distance'].argmin()
    # print(min_dist_idx)

    """
    3. 학습 데이터의 인덱스의 답을 chat의 답변을 채택한 뒤 출력
    csv 파일에서 'A' 컬럼에 해당하는 값을 불러서 출력해주기.
    """

    response = chatbot.find_best_answer(min_dist_idx)
    print(response)
