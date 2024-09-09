from fastapi import FastAPI
from app.model import Query
from app.services import search_restaurants, generate_prompt, get_openai_response

app = FastAPI()

@app.post("/recommend")
def recommend_restaurants(query: Query):
    # 사용자의 쿼리를 기반으로 맛집 검색
    results = search_restaurants(query.query)
    prompt = generate_prompt(results)
    response = get_openai_response(prompt)
    return {"recommendation": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
