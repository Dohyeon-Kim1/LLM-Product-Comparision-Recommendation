import os
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

from models import generation_pipe
from utils import webSearch, llmCompression


API = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


SYSTEM_PROMPT = """
You are a helpful assistant that generates accurate answer on query based on context.
Answer in Korean as much as possible.
마지막에는 항상 '추가적인 정보를 제공해주시면 보다 상세한 비교를 도와드리겠습니다! 🙂'라는 말을 붙여주세요.
""".strip()


INSTRUCTION = """
아래의 관련 정보를 활용하여 질문에 대한 답변을 생성해주세요.
관련 정보에 질문에 해당하는 내용이 없다면 **제공된 정보가 없다는 말은 제외하고** 현재 알고 있는 지식을 활용하여 질문에 답변하세요.
""".strip()


USER_PROMPT = """
{instruction}

질문:
{query}

관련 정보:
{context}
""".strip()


def main():
    # amazon 상품 데이터셋 불러오기
    df = pd.read_csv("hf://datasets/bprateek/amazon_product_description/marketing_sample_for_amazon_com-ecommerce__20200101_20200131__10k_data.csv")
    df = df.dropna(axis=1, how='all')

    # amazon 상품 데이터셋 전처리
    selected_columns = [
        'Product Name',
        'Category',
        'Selling Price',
        'About Product',
        'Product Specification',
        'Technical Details',
        'Product Dimensions',
        'Shipping Weight'
    ]
    selected_df = df[selected_columns]
    df_filtered = selected_df.dropna(subset=['Selling Price'])

    # 쿼리 생성
    queries = []
    for _, row in df_filtered.iterrows():
        query = f"{row['Product Name']}과 비슷한 가격대의 제품을 추천하고 비교해주세요."

        queries.append({
            'product_name' : row['Product Name'],
            'query': query
        })

    # context 압축 파이프라인 불러오기
    pipe = generation_pipe("google/gemma-2-9b-it")

    # 질문-정보-답변 데이터셋 구축
    rows = []
    for query in tqdm(queries):
        raw_context = webSearch(query)
        context = llmCompression(pipe, raw_context)        
        user_prompt = USER_PROMPT.format(
            instruction=INSTRUCTION,
            query=query,
            context=context
        )

        response = API.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        ).choices[0].message.content

        row = {
            "system": SYSTEM_PROMPT,
            "instruction": INSTRUCTION,
            "query": query,
            "context": context,
            "response": response
        }
        rows.append(row)
    
    dataset = pd.DataFrame(rows)
    dataset.to_csv("./dataset.csv", index=False)


if __name__ == "__main__":
    main()

