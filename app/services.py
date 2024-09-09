import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai

# Sentence Transformer 모델 로드
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# OpenAI API 키 설정
openai.api_key = "your_openai_api_key"

# FAISS index 불러오기 (미리 생성된 파일이 있다고 가정)
index = faiss.read_index("restaurant_index.faiss")

# 맛집 설명을 벡터화
def embed_texts(texts):
    return model.encode(texts)

# 벡터 검색
def search_restaurants(query, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    # 검색 결과를 리스트로 반환
    results = [restaurants[i] for i in indices[0]]
    return results

# 프롬프트 생성
def generate_prompt(restaurants):
    prompt = "Here are some recommended restaurants based on your query:\n"
    for restaurant in restaurants:
        prompt += f"{restaurant['title']}: {restaurant['description']} located at {restaurant['address']}\n"
    prompt += "Please let me know if you need more information."
    return prompt

# OpenAI 응답 생성
def get_openai_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()