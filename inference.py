from utils import webSearch, llmCompression


PROMPT = """<start_of_turn>user
You are an expert product recommendation AI assistant trained to:
1. Compare products objectively based on features and specifications.
2. Provide personalized recommendations considering user needs.
3. Explain the pros and cons of each product clearly.
4. Support recommendations with factual information.

[System]
당신은 제품 추천과 비교를 전문으로 하는 AI 어시스턴트입니다. 주어진 컨텍스트를 기반으로 정확한 답변을 한국어로 생성해주세요.
마지막에는 항상 '추가적인 정보를 제공해주시면 보다 상세한 비교를 도와드리겠습니다! 🙂'라는 말을 붙여주세요.

[Instruction]
제공된 정보를 기반으로 질문에 답변해주세요. 만약 질문과 관련된 내용이 없더라도, 그 사실을 언급하지 말고 현재 보유한 지식을 활용하여 답변해주세요.

[Question]
{query}

[Context]
설명: {context}
<end_of_turn>
<start_of_turn>model
""".strip()


def inference(pipe, query):
    """
    pipe: 허깅페이스의 파이프라인 인스턴스
    query: 사용자의 질문
    """

    # 텍스트 정리 함수(입력 텍스트 정리)
    def clean_text(text):
        return text.strip().replace('\n', ' ').replace('  ', ' ')  #strip(): 앞뒤 공백 제거
                                                                   #replace('\n', ' '): 줄바꿈 -> 공백 변환
                                                                   #replace('  ', ' '): 두 개 이상 연속 공백 하나로 줄임
    
    # 쿼리 전처리 및 RAG
    query = clean_text(query)
    raw_context = webSearch(query)
    preprocessed_context = llmCompression(pipe, raw_context)
    context = clean_text(preprocessed_context)

    # 프롬프트 작성
    prompt = PROMPT.format(query=query, context=context)

    # 모델 추론
    generated_text = pipe(
        prompt,
        max_new_tokens=2048,
        truncation=True,
        return_full_text=False  # Exclude input prompt
    )[0]["generated_text"]

    # 모델의 답변 부분만을 추출한 부분
    response = generated_text.split("<end_of_turn>")[0].strip()
    return response