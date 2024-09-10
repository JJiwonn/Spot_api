import requests
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

# 구글 플레이스 API 키 설정
GOOGLE_API_KEY = 'your_google_places_api_key'


def get_nearby_restaurants(location, radius=1000):
    """
    구글 플레이스 API로 주어진 위치와 반경 내의 레스토랑 정보를 가져옴.

    :param location: 위도, 경도 (예: "37.496456,127.029288")
    :param radius: 검색 반경 (미터 단위)
    :return: 레스토랑 리스트
    """
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={location}&radius={radius}&type=restaurant&key={GOOGLE_API_KEY}"

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        restaurants = data.get('results', [])
        return restaurants
    else:
        print(f"Error: {response.status_code}")
        return []


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
        prompt += f"{restaurant['name']}: located at {restaurant.get('vicinity', '주소 없음')}\n"
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


# 구글 플레이스 API를 통해 수집한 레스토랑을 FAISS 벡터화 후 검색
def get_restaurant_recommendations(query, location="37.496456,127.029288", radius=1000):
    # 1. 구글 플레이스 API에서 레스토랑 데이터를 가져옴
    nearby_restaurants = get_nearby_restaurants(location, radius)

    # 2. 레스토랑 이름 벡터화
    restaurant_names = [restaurant['name'] for restaurant in nearby_restaurants]
    restaurant_embeddings = embed_texts(restaurant_names)

    # 3. FAISS 인덱스에 추가 (필요한 경우)
    index.add(np.array(restaurant_embeddings))

    # 4. 쿼리를 벡터화하고 레스토랑 검색
    results = search_restaurants(query)

    # 5. 프롬프트 생성
    prompt = generate_prompt(results)

    # 6. OpenAI 응답 생성
    response = get_openai_response(prompt)

    return response


# 예시 호출
if __name__ == '__main__':
    query = "best Italian restaurant"
    recommendations = get_restaurant_recommendations(query)
    print(recommendations)
