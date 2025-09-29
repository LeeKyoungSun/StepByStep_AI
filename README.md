# RAG Stack (Embeddings → FAISS → RAG → Moderation)

한국어 청소년 성교육 챗봇을 위한 최소구성 RAG 스택입니다.

## 구성
- 임베딩: OpenAI `text-embedding-3-small`(기본). 필요시 `text-embedding-3-large`로 교체 가능
- 검색: FAISS (cosine = L2 정규화 + Inner Product)
- 생성: OpenAI `gpt-4.1-mini`(기본), 어려운 질문은 상향 교체 가능
- 모더레이션: `omni-moderation-latest` (입출력 이중 필터)

## 빠른 시작
1) Python 3.10+ 권장, 가상환경 생성 후 패키지 설치
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
2) 환경변수 준비
```bash
cp .env.example .env
# .env 파일에서 OPENAI_API_KEY, 모델명 확인/수정
```
3) 데이터 폴더에 OCR 텍스트 파일 놓기
```
data/
  your_document_1.txt
  your_document_2.txt
```
4) 인덱스 빌드
```bash
python build_index.py
```
5) 서버 실행
```bash
uvicorn serve_rag:app --host 0.0.0.0 --port 8000 --reload
```
6) 질의 테스트
```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"콘돔 올바르게 사용하는 방법 알려줘"}'
```

## 폴더
- `data/` : txt 원문
- `index/` : FAISS 인덱스(`faiss.index`), 메타데이터(`meta.jsonl`), 설정(`index_info.json`)

## 참고
- 청크 크기/겹침, k값 등은 `build_index.py`, `utils/chunker.py`에서 조절하세요.
- 반말/공감형 톤은 `utils/generator.py`의 시스템 프롬프트에서 관리합니다.
