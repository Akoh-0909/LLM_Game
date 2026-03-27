# Humanity Check: Protocol 404
Introduction

---

## 1. 게임 소개

게임의 배경은 서기 2126년, 기계가 지배하는 세상입니다.  
유저는 AI 면접관 앞에서 자신이 기계가 아닌 인간임을 증명해야 합니다.  
논리적인 답변이 아니라, 감정의 불완전함과 모순으로 인간임을 증명하는 게임입니다.

총 5라운드, 목숨 3개, 각 질문당 5초 제한이 있으며  
사이코패스 판정을 받으면 즉시 탈락합니다.

---

## 2. 전체 기술 구조 및 학습 출처

이 게임은 수업에서 배운 7가지 기술을 모두 활용해서 구성하였음.

[1] RAG + ChromaDB:심리학 문서를 벡터로 저장, 판정 시 관련 문서 검색해서 LLM에 제공
[1] OpenAI 임베딩: 심리학 문서를 숫자 벡터로 변환
[2] 프롬프트 엔지니어링 (CoT): 공감→도덕추론→감정진정성 3단계로 LLM이 단계적으로 분석하도록 유도
[2] Function Calling: 판정 결과를 HUMAN/MACHINE/PSYCHO + 점수로 구조화해서 받아옴
[3] LangGraph: 라운드/목숨/점수/종료 등 게임 상태 관리 및 흐름 제어
[4] gTTS:면접관 질문, 판정결과 - 음성으로 출력
[5] STT: 유저음성답변 - 텍스트로 변환
[7] Streamlit - 웹기반게임UI 

---

### Step 1. RAG + ChromaDB + 임베딩

가장 먼저 사이코패스 분류 기준을 정의하기 위해 심리학 참조 문서를 준비.
공감, 도덕적 추론, 감정의 복잡성, 사이코패스 판별 기준 등  
총 12개의 심리학 문서 및 논문내용을 발췌하여 코드 안에 리스트로 직접 작성하였음
RAG: analyze_answer()함수안에서 활용
RAG로 관련문서 검색 -> 검색된 문서를 프롬프트 {context} 자리에 삽입

이 문서들은 수업에서 배운 대로 `text-embedding-3-small` 모델로 임베딩하여  
ChromaDB에 저장함.

```python
# 04_embeddings.ipynb에서 배운 패턴
def get_embedding(text: str) -> list:
    return client_openai.embeddings.create(
        model='text-embedding-3-small', input=text
    ).data[0].embedding
```
** 처음에는 chromadb.Client()를 사용하였으나 데이터 저장이 되지 않아 매번 셀4번을 재실행하여 임베딩 API비용이 발생하고 임베딩 생성 시간이 소요되므로 PersistentClient 사용으로 변경하였음 **

ChromaDB는 `PersistentClient`를 사용했는데,  
이유는 한 번 저장해두면 최초 1회만 비용 발생하고 노트북을 재실행해도 데이터가 유지되고  
Streamlit app.py에서도 동일한 DB를 공유해서 사용할 수 있기 때문임.

```python
# 01_chroma_db.ipynb에서 배운 패턴
chroma_client = chromadb.PersistentClient(path='./chroma_db')
collection = chroma_client.create_collection(name='psychology')
collection.add(
    ids=[...], embeddings=[...], documents=[...], metadatas=[...]
)
```

RAG 검색 함수는 유저 답변을 분석할 때 관련 심리학 문서를 검색해서  
LLM 프롬프트에 컨텍스트로 제공하였는데, 
이렇게 하면 LLM이 심리학 이론에 근거해서 판정을 내릴 수 있을거라 판단함.

---

### Step 2. 프롬프트 엔지니어링 + Function Calling


판정 엔진: 두 가지 학습 내용을 결합함.

1) **CoT(Chain of Thought) 프롬프트** 
`03-1_rag_cot.ipynb`에서 배운 대로 LLM이 단계적으로 사고하도록 유도하였음.

```
[CoT 분석 단계]
1. 공감 지수 (0~33): 감정 반응이 자연스럽고 구체적인가?
2. 도덕 추론 지수 (0~33): 갈등과 고민의 흔적이 있는가?
3. 감정 진정성 지수 (0~33): 모순적이고 불완전할수록 높은 점수
```

2) **with_structured_output** 
Pydantic BaseModel과  
`llm.with_structured_output()`을 사용해서 판정 결과를 구조화

```python
class HumanityVerdict(BaseModel):
    verdict: Literal['HUMAN', 'MACHINE', 'PSYCHO']
    empathy_score: int
    moral_score: int
    authenticity_score: int
    ...

analysis_chain = analysis_prompt | llm.with_structured_output(HumanityVerdict)
```

총점은 LLM에 맡기지 않고 세 점수를 코드에서 직접 합산함  
-> LLM이 각 점수와 총점을 불일치하게 반환할 수도..

```python
total = v.empathy_score + v.moral_score + v.authenticity_score
```

---

### Step 3. LangGraph 게임 상태 관리


게임의 라운드, 목숨, 점수, 종료 여부 등 상태 관리를 위해  
수업에서 배운 LangGraph를 활용

```python

class GameState(TypedDict):
    round: int
    lives: int
    total_score: int
    game_over: bool
    messages: Annotated[List, add_messages]
    ...
```

4개의 노드(question → analyze → update → report)와  
조건부 엣지(should_continue)로 게임 흐름을 설계

```python
workflow.add_conditional_edges('update', should_continue,
    {'question': 'question', 'report': 'report'})
```

사이코 판정 시 즉시 탈락, 기계 판정 시 목숨 감소,  
인간 판정 시 다음 라운드로 진행하는 분기를 update_node에서 처리.

---

### Step 4. gTTS 음성 출력


면접관의 질문과 판정 결과를 음성으로 출력.  
gTTS 활용

```python

tts = gTTS(text=text, lang='ko')
tts.save(f.name)
display(Audio(f.name, autoplay=True))
```

---

### Step 5. STT 음성 입력


유저가 텍스트 대신 음성으로 답변할 수 있도록 STT를 구현.  
마이크가 없는 환경에서도 동작할 수 있도록  
마이크 장치 사전 체크 로직 추가

```python

recognizer.recognize_google(audio, language='ko-KR')
```

---

### Step 6~7. 게임 실행 및 Streamlit UI

Jupyter에서 바로 플레이할 수 있는 텍스트 기반 게임 루프와  
Streamlit 기반 UI를 함께 제공.

Streamlit app.py -> 노트북에서 생성한 `./chroma_db`를  
`get_collection()`으로 불러와서 사용. 
=> 반드시 노트북 Step 1을 먼저 실행한 후 Streamlit을 실행해야 함.

---

## 3. 핵심 설계 포인트 

**왜 RAG를 사용했는가?**  
LLM이 사이코패스 판별을 할 때 심리학 이론에 근거하도록 하기 위해서.  
RAG 없이 판정하면 LLM이 임의적으로 판단할 수 있음.

**왜 with_structured_output을 사용했는가?**  
판정 결과(HUMAN/MACHINE/PSYCHO)와 점수를 정형화된 형태로 받아야  
게임 로직에서 안정적으로 처리할 수 있기 때문.

**왜 PersistentClient를 사용했는가?**  
임베딩 API 호출은 비용과 시간이 발생함.  
PersistentClient를 쓰면 최초 1회만 생성하고  
이후에는 저장된 DB를 재사용할 수 있음 + Streamlit app.py와 동일한 DB를 공유 가능.

**왜 CoT 프롬프트를 사용했는가?**  
단계적분석을 통해 더 정교하고 근거 있는 판정을 하기 위해. 

---


