import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MOD_MODEL = "omni-moderation-latest"

def check_text_safety(text: str) -> dict:
    resp = client.moderations.create(
        model=MOD_MODEL,
        input=text
    )
    # OpenAI Moderation API returns categories & flagged
    result = resp.results[0]
    return {
        "flagged": result.flagged,
        "categories": result.categories,
        "scores": result.category_scores if hasattr(result, "category_scores") else {}
    }

SAFE_FALLBACK = (
    "미안해. 이 주제는 안전과 정책을 위해 자세히 다룰 수 없어.\n"
    "대신 성·건강 관련 정확한 정보는 신뢰할 수 있는 공식 자료를 참고하고,\n"
    "긴급하거나 불편한 상황이면 112 또는 가까운 도움 기관에 바로 연락해줘."
)
