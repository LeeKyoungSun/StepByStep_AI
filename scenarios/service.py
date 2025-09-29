# SCSC/scenario/service.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from collections import Counter, deque
from pathlib import Path
import json, random, re, os

from openai import OpenAI

from SCSC.utils.prompts import SCENARIO_PROMPT, USER_TMPL
from SCSC.scenarios.keyword_rules import match_keywords

from SCSC.utils.faiss_store import FaissStore
from SCSC.utils.bm25_store import BM25Store

# service.py 최상단(또는 prompts/constants 모듈)에 교체
CONCEPT_MAP = {
    "성병": [
        # 정의/특징
        "HPV 정의", "HPV 유형", "HPV 백신",
        "헤르페스 특징", "헤르페스 증상", "헤르페스 재발",
        "클라미디아 증상", "클라미디아 무증상 가능성",
        "임질 증상",
        "매독 1기 특징", "매독 2기 특징", "매독 3기 특징",
        "HIV 전파", "HIV 검사", "HIV 치료",
        "B형간염 전파", "B형간염 예방접종",
        "트리코모나스 특징", "트리코모나스 치료",

        # 전파/예방/검사
        "무증상 가능성", "잠복기 개념", "검사 권장 시점",
        "성병 감염 경로", "성병 전파 방식",
        "성병 검사 방법", "성병 검사 주기",
        "자가키트 활용", "익명검사 활용",
        "파트너 통보", "동시 치료", "재감염 가능성", "치료 완료 기준",

        # 예방 수단
        "콘돔의 성병 예방 효과",
        "구강 성교 예방(덴탈댐)", "항문 성교 예방(콘돔)",
        "백신으로 예방 가능한 감염",

        # 오해 교정
        "목욕탕 전파 오해", "화장실 좌변기 전파 오해",
        "키스만으로 전파 오해", "항생제 남용 위험", "항생제 내성 위험",
    ],

    "피임": [
        # 배리어/행동
        "콘돔 개봉", "콘돔 꼭지 공기 빼기", "콘돔 착용 순서", "콘돔 탈착",
        "콘돔 보관법", "콘돔 파손 원인",
        "질외사정 실패율", "질외사정 전분비액 위험",

        # 응급/호르몬/장치
        "사후피임약 복용 시점", "사후피임약 효과", "사후피임약 부작용",
        "경구피임약 복용법", "경구피임약 복용 누락 대처", "경구피임약 부작용",
        "IUD 구리 장단점", "IUD 호르몬 장단점", "IUD 부작용",
        "피임 패치 작용", "피임 패치 대상",
        "피임 주사 작용", "피임 주사 대상",
        "피임 임플란트 작용", "피임 임플란트 대상",
        "질정 사용법", "질정 실패율",
        "다이어프램 사용법", "다이어프램 실패율",

        # 실패/상담
        "피임 실패 시 응급피임", "피임 실패 시 임신 가능성 평가", "피임 실패 상담",
        "피임과 성병 예방의 차이", "이중 보호",
        "피임 의사소통 전략",
    ],

    "생리": [
        "월경 주기", "배란", "개인차 원인", "불규칙 원인",
        "가임기 계산 한계", "가임기 계산 오해",
        "월경통 자기관리", "월경통 약물", "월경통 경고 신호",
        "PMS 특징", "PMDD 특징", "PMS/PMDD 대처",
        "생리대 사용", "탐폰 사용", "생리컵 사용",
        "교체 주기", "위생 관리",
        "스팟팅 원인", "주기 변화 원인", "스트레스 영향", "체중 변화 영향", "약물 영향",
        "초경 안내", "사춘기 변화",
        "수영 시 용품 선택", "체육 시 용품 선택",
        "과다 월경 상담 기준", "과소 월경 상담 기준",
    ],

    "경계/동의": [
        "동의 원칙: 자유", "동의 원칙: 명확성", "동의 원칙: 구체성", "동의 원칙: 가역성",
        "취중 동의 무효", "압박 관계 동의 무효", "권력관계 동의 무효",
        "경계 설정 방법", "의사표현 문장 예시",
        "거절 뒤 대화", "관계 존중",
        "디지털 동의: 사진 촬영", "디지털 동의: 영상 공유",
        "동의의 지속성", "동의의 철회",
    ],

    "관계/의사소통": [
        "감정 인식", "감정 표현", "경청 스킬",
        "나-메시지", "비난 대신 구체적 요청",
        "갈등 해결: 사실 분리", "갈등 해결: 감정 분리", "갈등 해결: 요청 분리",
        "연애 의사결정", "연애 상호 존중",
        "개인정보 공유 범위", "비밀보장",
        "질문 스킬", "확인 스킬", "확증 편향 줄이기",
    ],

    "온라인/디지털": [
        "디지털 성범죄: 불법촬영", "디지털 성범죄: 유포", "디지털 성범죄: 협박",
        "사진 요구 거절 문장", "사진 요구 차단", "증거 보존",
        "신고 112", "상담 1366", "디지털 성범죄 지원단",
        "2단계 인증", "비밀번호 관리",
        "디지털 동의 기본", "저작권 기본", "초상권 기본",
        "유해물 노출 대처", "유해물 신고",
    ],

    "임신/출산": [
        "임신 가능성", "가임기 오해 바로잡기",
        "임신 테스트기 시점", "임신 테스트기 판독",
        "임신 초기 증상", "임신 확인 절차",
        "임신중절 정보 접근", "임신중절 상담 경로",
        "의료기관 찾기", "비밀보장",
        "임신 의사결정 지원", "임신 안전",
    ],

    "건강/상담": [
        "학교 보건실 활용", "보건교사 활용",
        "청소년 친화 의료기관 찾기",
        "비밀보장", "동반자 동의",
        "불안과 성", "우울과 성", "정신건강과 성의 관계",
        "상담소 이용", "헬프라인 112", "헬프라인 1366", "정신건강 위기 대응",
        "상담 준비: 증상 기록", "상담 준비: 질문 리스트",
    ],

    "신체 변화": [
        "사춘기 2차 성징", "개인차 존중",
        "음경 이해", "고환 이해", "포경 이해", "몽정 이해", "발기 이해",
        "유방 발달", "브라 선택",
        "체모 변화", "목소리 변화", "피부 변화 관리",
        "신체 이미지", "자기존중감",
    ],

    "외모/자기이미지": [
        "체중과 건강", "체형과 건강",
        "다이어트 오해 바로잡기",
        "여드름 관리", "피부 관리 기본",
        "미디어 보정 인식", "미디어 필터 인식",
        "외모 괴롭힘 대처",
    ],

    "자위/욕구": [
        "성적 욕구 정상성", "자위 정상성",
        "자위와 건강 오해",
        "프라이버시", "위생", "디지털 안전",
        "콘텐츠 선택", "경계 설정",
    ],
}

from collections import deque

SCENARIO_BACKGROUNDS = [
    "수업 끝나고 복도에서 대화 중",
    "동아리 활동 쉬는 시간",
    "단체 채팅에서 의견을 나누는 중",
    "공원 벤치에서 이야기하는 중",
    "보건실 상담 대기 중",
    "온라인 메신저 대화 중",
    "급식 줄에서 잡담 중",
    "조별 과제 회의 중",
    "등굣길 버스 안",
    "도서관 자습 중",
    "체육 시간 팀 활동 전",
    "학급 게시판 앞",
    "학교 축제 준비 중",
    "청소년상담복지센터 대기실",
    "보건소 예약 전화 전",
    "주말 스터디 카페",
]

SCENARIO_ENDINGS = [
    "너라면 어떻게 할래?",
    "지금 선택할 행동은 무엇일까?",
    "어떤 말부터 꺼낼래?",
    "먼저 확인해야 할 것은 무엇일까?",
    "누구와 상의해볼까?",
    "가장 안전한 선택은 무엇일까?",
    "네가 취할 수 있는 다음 한 걸음은?",
    "상대를 존중하면서 뭐라고 말하겠어?",
]

CONCEPT_FORMS = [
    "{topic}는 무엇일까?",
    "다음 중 {topic}의 특징으로 옳은 것은?",
    "{topic}에 대한 설명으로 맞는 것을 골라.",
    "{topic} 예방 또는 관리 방법으로 올바른 것은?",
]

# 중복 방지: 최근 N개 문제 문장 캐시
RECENT_MAX = 20

@dataclass
class Config:
    index_root: str = "SCSC/indexes"  # window/ qna 루트 상위
    topk: int = 6
    max_context_chars: int = 1600
    gen_model: str = "gpt-4o-mini"


class ScenarioService:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.faiss = self._try_load_faiss()
        self.bm25 = self._try_load_bm25()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._pool = self._load_pool()
        self._keywords = None
        self._recent_questions = deque(maxlen=RECENT_MAX)
        self._topic_cycle: Dict[str, List[str]] = {}

    # ---------- 상태 유틸 추가 --------
    # 톤 통일(반말) 간단 정규화
    @staticmethod
    def _unify_register(text: str) -> str:
        if not isinstance(text, str): return text
        t = re.sub(r"요[.!?]?$", "", text)  # 문장 끝 "~요" 살짝 제거
        t = t.replace("하십시오", "해").replace("하세요", "해").replace("해주세요", "해줘")
        return re.sub(r"\s+", " ", t).strip()

    # 질문 유사도 가드(너무 비슷하면 True)
    @staticmethod
    def _too_similar(a: str, b: str, th: float = 0.7) -> bool:
        ta = set(a.replace(",", " ").replace(".", " ").split())
        tb = set(b.replace(",", " ").replace(".", " ").split())
        if not ta or not tb: return False
        j = len(ta & tb) / max(1, len(ta | tb))
        return j >= th

    def _push_recent(self, q: str) -> bool:
        # 중복이면 False
        for prev in self._recent_questions:
            if self._too_similar(prev, q):
                return False
        self._recent_questions.append(q)
        return True

    def _next_concept_topic(self, kw: str) -> Optional[str]:
        pool = CONCEPT_MAP.get(kw or "", [])
        if not pool: return None
        # 키워드별 로테이션(한 번 섞어두고 pop)
        if kw not in self._topic_cycle or not self._topic_cycle[kw]:
            arr = pool[:]
            random.shuffle(arr)
            self._topic_cycle[kw] = arr
        return self._topic_cycle[kw].pop()

    def _scenario_hint(self) -> str:
        bg = random.choice(SCENARIO_BACKGROUNDS)
        end = random.choice(SCENARIO_ENDINGS)
        return f"[배경] {bg}\n[마무리 질문] {end}"

    def show_sources(self, item):
        """퀴즈 아이템의 근거 문단을 풀텍스트로 보여줌"""
        out = []
        for src in item.get("sources", []):
            for s in self._pool:
                if s["source"] == src["source"] and s["chunk_id"] == src["chunk_id"]:
                    out.append(f"[{src['source']} #{src['chunk_id']}] {s['text']}")
                    break
        return "\n".join(out) if out else "(근거 없음)"

    # ---------- 로드/빌드 ----------
    def _try_load_faiss(self):
        root = Path(self.cfg.index_root)
        candidates = list(root.glob("**/*_mac")) + list(root.glob("**/*_window"))
        for d in candidates:
            try:
                return FaissStore.load(str(d))
            except Exception:
                continue
        return None

    def _try_load_bm25(self):
        root = Path(self.cfg.index_root)
        candidates = list(root.glob("**/*_mac")) + list(root.glob("**/*_window"))
        for d in candidates:
            try:
                return BM25Store.load(str(d))
            except Exception:
                continue
        return None

    def _load_pool(self):
        """인덱스에서 전체 스니펫 풀 1회 구축"""
        pool: List[Dict[str, Any]] = []
        for p in Path(self.cfg.index_root).glob("**/meta.json"):
            arr = json.loads(p.read_text(encoding="utf-8"))
            for row in arr:
                txt = row.get("text") or row.get("chunk_text") or ""
                if not txt:
                    continue
                kws = row.get("keywords") or match_keywords(txt)
                pool.append({
                    "text": " ".join(txt.split()),
                    "source": row.get("src") or row.get("source") or p.parent.name,
                    "chunk_id": row.get("chunk_id"),
                    "keywords": kws,
                })
        return pool

    # ---------- 검색 ----------
    def search(self, query: str, topk: int) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if self.faiss:
            results += self.faiss.search(query, top_k=topk)
        if self.bm25:
            results += self.bm25.search(query, top_k=topk)
        if not results:
            return []
        # 간단 RRF
        def k(s): return f"{s.get('source')}#{s.get('chunk_id')}"
        scored: Dict[str, float] = {}
        for rank, s in enumerate(results, start=1):
            scored[k(s)] = scored.get(k(s), 0.0) + 1.0 / (60 + rank)
        uniq = {k(s): s for s in results}
        ranked = sorted(uniq.values(), key=lambda s: scored[k(s)], reverse=True)
        return ranked[:topk]

    def random_snippets(self, topk: int) -> List[Dict[str, Any]]:
        if not self._pool:
            return []
        return random.sample(self._pool, k=min(topk, len(self._pool)))

    # ---------- 키워드 ----------
    def keywords(self, limit: int = 40):
        # 표시 순서를 고정하고 싶어 튜플로 관리
        WHITELIST = (
            "피임", "생리", "연애", "외모", "신체 변화",
            "젠더", "관계/의사소통", "경계/동의", "온라인/디지털",
            "성병/검사", "임신/출산", "자위/욕구", "건강/상담",
        )

        c = Counter()
        for s in self._pool:
            for kw in s.get("keywords", ()):  # 안전하게
                c[kw] += 1

        # 화이트리스트 순서대로만 뽑되, 실제로 카운트가 있는 것만 노출
        ordered = [(k, c[k]) for k in WHITELIST if k in c]

        # 전혀 없으면(태깅/매칭 안 되어 있을 때) 전체 상위로 폴백
        if not ordered:
            ordered = sorted(c.items(), key=lambda x: x[1], reverse=True)

        return [{"keyword": k, "count": v} for k, v in ordered[:limit]]

    def pick_by_keyword(self, keyword: str, topk: int):
        cand = [s for s in self._pool if keyword in s.get("keywords", [])]
        if not cand: return []
        random.shuffle(cand)
        if self.faiss:
            # '상황' 단어 제거 → 상황형 편향 완화
            query = f"{keyword} 원칙 개념 예방 검사 특징 사례"
            ranked = self.faiss.search(query, top_k=topk * 3)
            allow = {(s["source"], s.get("chunk_id")) for s in cand}
            ranked = [r for r in ranked if (r.get("source"), r.get("chunk_id")) in allow]
            random.shuffle(ranked)
            if ranked:
                return ranked[:topk]
        return cand[:topk]

        # ───────── 품질 체크 유틸 ─────────

    def _concept_snippets(self, keyword: str, topic: str, topk: int) -> List[Dict[str, Any]]:
        def _variants(t: str) -> List[str]:
            t2 = re.sub(r"[\(\)]", " ", t)
            toks = [t, t2]
            if "HPV" in t.upper(): toks += ["HPV", "인유두종", "인유두종바이러스"]
            if "HIV" in t.upper(): toks += ["HIV", "AIDS", "에이즈"]
            return list(dict.fromkeys([re.sub(r"\s+", " ", x).strip() for x in toks if x.strip()]))

        pats = [re.compile(re.escape(v), re.IGNORECASE) for v in _variants(topic)]
        hits = []
        for s in self._pool:
            if keyword and keyword not in (s.get("keywords") or []): continue
            txt = s.get("text") or ""
            if any(p.search(txt) for p in pats):
                hits.append(s)
                if len(hits) >= topk: break
        if len(hits) >= max(2, topk // 2):
            random.shuffle(hits);
            return hits[:topk]

        q = f"{topic} 정의 특징 전파 경로 예방 검사 설명 근거"
        found = self.search(q, topk=topk * 2)
        if keyword and found:
            f2 = [r for r in found if keyword in (r.get("keywords") or [])]
            found = f2 or found
        random.shuffle(found)
        return (found or hits or self.random_snippets(topk))[:topk]

    @staticmethod
    def _normalize_choices(choices: List[str]) -> List[str]:
        norm = []
        for c in choices:
            if not isinstance(c, str): continue
            cc = re.sub(r"^\s*[A-Da-d]\s*[\.\):]\s*", "", c).strip()  # "A. " 제거
            cc = re.sub(r"\s+", " ", cc)
            if cc:
                norm.append(cc)
        # 중복 제거
        seen, out = set(), []
        for c in norm:
            if c not in seen:
                out.append(c);
                seen.add(c)
        return out

    @staticmethod
    def _looks_bad_choices(choices: List[str]) -> bool:
        if len(choices) != 4:
            return True
        # 너무 짧거나 중복, 극단 키워드만 나열인지 검사
        if any(len(c.strip()) < 8 for c in choices):
            return True
        norm = [re.sub(r"\s+", " ", c.strip()) for c in choices]
        if len(set(norm)) < 4:
            return True
        bad_phrases = ["무시하고", "강제로", "즉시 관계", "피임 없이", "아무 준비 없이"]
        if sum(any(p in c for p in bad_phrases) for c in norm) >= 3:
            return True
        return False

    @staticmethod
    def _contains_english_name(text: str) -> bool:
        # 영문 이름/로마자 탐지(간단 버전)
        return bool(re.search(r"\b[A-Z][a-z]{2,}\b", text or ""))

    def make_quiz_item(
            self,
            keyword: Optional[str],
            snips: List[Dict[str, Any]],
            force_type: Optional[str] = None,  # "concept" | "situation" | None
            concept_topic: Optional[str] = None,  # ex) "HPV(인유두종바이러스)"
    ):
        # 1) 유형/주제 확정
        qtype = force_type or "situation"
        topic = concept_topic or (keyword or "핵심 개념")

        # 2) 컨텍스트/추가 힌트 구성
        if qtype == "concept":
            # ✅ 개념형은 사전식 정의: 인덱스 컨텍스트 사용 안 함(모델 지식 신뢰)
            snips = []
            context = ""
            try:
                form = random.choice(CONCEPT_FORMS).format(topic=topic)
            except Exception:
                form = f"{topic}는 무엇인가?"
            extra_hint = (
                "[출제 형태] type=concept\n"
                f"[개념 주제] {topic}\n"
                f"[질문 형식 예시] {form}\n"
                "- 보기 4개: 정확한 정의/특징(정답) + 불완전 + 오해 + 무관.\n"
                "- 질문은 시험문장 형태로 끝내고, '함께 확인해보자/같이 알아보자' 등 권유형 접미는 쓰지 말 것.\n"
            )
        else:
            # 상황형은 인덱스 기반 컨텍스트 유지
            context = self._mk_context(snips)
            try:
                extra_hint = (
                        self._scenario_hint() +
                        "\n[출제 형태] type=situation\n"
                        "- 보기 4개: 정답(근거 기반 안전행동) + 불완전 + 오해 + 부적절.\n"
                )
            except Exception:
                extra_hint = (
                    "[출제 형태] type=situation\n"
                    "- 보기 4개: 정답(근거 기반 안전행동) + 불완전 + 오해 + 부적절.\n"
                )

        # 3) prompts.py 톤 프리셋 지원(신/구 템플릿 모두 동작)
        tone = "친근반말"
        try:
            from SCSC.utils.prompts import TONE_PRESETS
            user = USER_TMPL.format(
                qtype=qtype, tone=tone, keyword=keyword or "(랜덤)",
                tone_block=TONE_PRESETS.get(tone, ""), context=context,
            ) + "\n" + extra_hint
        except Exception:
            tone_block = (
                "[말투 가이드]\n"
                "- 친구에게 말하듯 따뜻하고 존중하는 반말.\n"
                "- 비난/조롱 금지, 정보와 근거 중심.\n"
                "- 문장 간결(1~2절), 권유형 접미(함께/같이 ~하자) 금지.\n"
            )
            user = USER_TMPL.format(keyword=keyword or "(랜덤)", context=context) \
                   + "\n" + tone_block + "\n" + extra_hint

        # 4) 모델 호출 & 보정 루프
        data = None
        for attempt in range(3):
            resp = self.client.chat.completions.create(
                model=self.cfg.gen_model,
                temperature=0.2 if attempt == 0 else 0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SCENARIO_PROMPT},
                    {"role": "user", "content": user},
                ],
            )
            data = json.loads(resp.choices[0].message.content)

            # --- choices 정리 ---
            choices = data.get("choices", [])
            try:
                choices = self._normalize_choices(choices)
            except Exception:
                norm = []
                for c in (choices or []):
                    if not isinstance(c, str): continue
                    cc = re.sub(r"^\s*[A-Da-d]\s*[\.\):]\s*", "", c).strip()
                    cc = re.sub(r"\s+", " ", cc)
                    if cc: norm.append(cc)
                seen, tmp = set(), []
                for c in norm:
                    if c not in seen:
                        tmp.append(c);
                        seen.add(c)
                choices = tmp
            while len(choices) < 4:
                choices.append("추가 보기가 필요합니다.")
            data["choices"] = choices[:4]

            # --- 정답 인덱스/라쇼날/타입 보정 ---
            ai = data.get("answer_index")
            if not isinstance(ai, int) or not (0 <= ai < 4):
                data["answer_index"] = 0
            if not data.get("rationale"):
                data["rationale"] = (
                    "정답은 정의/특징·예방 등 정확한 설명이야. "
                    "다른 보기는 오해·불완전·무관한 설명이야."
                )
            data["type"] = qtype

            # --- 소스 처리: 개념형은 비움, 상황형만 인덱스 근거 유지 ---
            if qtype == "concept":
                data["sources"] = []
                data["evidence"] = "model_knowledge"  # (선택) UI용 플래그
            else:
                if not data.get("sources"):
                    data["sources"] = [
                        {"source": s.get("source"), "chunk_id": s.get("chunk_id")}
                        for s in snips[:2]
                    ]

            # --- 톤/표현 정리 ---
            try:
                data["question"] = self._unify_register(data.get("question", ""))
                data["choices"] = [self._unify_register(c) for c in data["choices"]]
                data["rationale"] = self._unify_register(data.get("rationale", ""))
            except Exception:
                pass

            if data.get("type") == "concept":
                data["question"] = re.sub(
                    r"(함께\s*확인해보자|같이\s*알아보자|함께\s*알아보자)\s*\.?$",
                    "", data["question"]
                ).strip()

            # --- 품질/타입 가드 ---
            bad = (
                    self._contains_english_name(data.get("question", "")) or
                    any(self._contains_english_name(c) for c in data["choices"]) or
                    self._looks_bad_choices(data["choices"]) or
                    (qtype == "concept" and data.get("type") != "concept")
            )
            if bad:
                if qtype == "concept":
                    user += "\n[개선요청] 개념 질문(type='concept')으로 정의/특징/예방/검사 중 하나를 정확히 묻고 4지선다로 작성해."
                    user += "\n[금지] '함께/같이 ~해보자' 같은 권유형 접미는 쓰지 마."
                else:
                    user += "\n[개선요청] 배경·갈등·질문을 분명히 하고, 보기들은 서로 다른 전략으로 구체화해."
                continue

            # 최근 질문 중복 방지
            try:
                if not self._push_recent(data["question"]) and attempt < 2:
                    if qtype == "concept":
                        form2 = (random.choice(CONCEPT_FORMS).format(topic=topic)
                                 if 'CONCEPT_FORMS' in globals() else f"{topic}의 특징으로 옳은 것은?")
                        user += f"\n[개선요청] 같은 의미지만 다른 문장/형식으로 재구성하고, 질문 형식을 '{form2}'로 바꿔."
                    else:
                        user += "\n[개선요청] 배경/표현을 바꿔 유사도를 낮춰."
                    continue
            except Exception:
                pass

            break  # 성공

        # 5) 보기 섞고 정답 인덱스 재계산
        correct_choice = data["choices"][data["answer_index"]]
        random.shuffle(data["choices"])
        data["answer_index"] = data["choices"].index(correct_choice)
        data["answer_letter"] = ["A", "B", "C", "D"][data["answer_index"]]
        return data

    # ---------- 생성 ----------
    def _mk_context(self, snips: List[Dict[str, Any]]) -> str:
        buf, cur = [], 0
        for s in snips:
            t = " ".join((s.get("text") or "").split())
            if not t:
                continue
            if cur + len(t) > self.cfg.max_context_chars:
                t = t[: max(0, self.cfg.max_context_chars - cur)]
            buf.append(f"- ({s.get('source','')}, chunk#{s.get('chunk_id')}) {t}")
            cur += len(t)
            if cur >= self.cfg.max_context_chars:
                break
        return "\n".join(buf)

    def make_quiz(self, mode: str, keyword: Optional[str], n: int = 5):
        out: List[Dict[str, Any]] = []

        for i in range(max(1, n)):
            # 스니펫 선택
            if mode == "by_keyword" and keyword:
                snips = self.pick_by_keyword(keyword, self.cfg.topk)
                if not snips:
                    expand = {"피임": ["성병/검사", "경계/동의"],
                              "생리": ["신체 변화", "건강/상담"],
                              "연애": ["관계/의사소통", "경계/동의"]}
                    for k2 in expand.get(keyword, []):
                        snips = self.pick_by_keyword(k2, self.cfg.topk)
                        if snips: break
                if not snips:
                    snips = self.random_snippets(self.cfg.topk)
                kw = keyword
            else:
                snips = self.random_snippets(self.cfg.topk)
                kw = (snips and snips[0].get("keywords") and random.choice(snips[0]["keywords"])) or "랜덤"

            # 유형 분배
            force_type = "concept" if (i % 2 == 1) else "situation"
                # 개념형 토픽: 로테이션으로 매번 다르게

            concept_topic = None
            if force_type == "concept":
                concept_topic = self._next_concept_topic(kw or keyword or "")

            item = self.make_quiz_item(kw, snips, force_type=force_type, concept_topic=concept_topic)
            if isinstance(item, dict) and "choices" in item and "answer_index" in item:
                out.append(item)
        return out
