from fastapi import FastAPI
from app.routers import quiz, health

app = FastAPI(
    title="StepByStep AI API",
    version="0.1.0",
    description="시나리오 퀴즈/헬스체크 API",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

# 라우터 등록
app.include_router(quiz.router, prefix="/api/quiz", tags=["quiz"])
app.include_router(health.router, tags=["health"])