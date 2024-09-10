from fastapi import FastAPI
from app.model import Query
from app.services import get_restaurant_recommendations

app = FastAPI()

@app.post("/recommend")
def recommend_restaurants(query: Query):
    # 사용자의 쿼리를 기반으로 맛집 검색 (구글 플레이스 API 사용)
    response = get_restaurant_recommendations(query.query)
    return {"recommendation": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
