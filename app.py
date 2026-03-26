import os, time, tempfile, random
import streamlit as st
from dotenv import load_dotenv
from gtts import gTTS
import chromadb
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal, TypedDict, Annotated, List
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage
from langgraph.graph.message import add_messages

load_dotenv()
st.set_page_config(page_title="Humanity Check: Protocol 404", page_icon="🤖", layout="centered")
st.markdown(
    "<style>.stApp{background:#0a0a0a;color:#00ff41;font-family:monospace}"
    "h1,h2,h3{color:#00ff41}"
    ".stButton>button{background:#001a00;color:#00ff41;border:1px solid #00ff41;font-family:monospace}"
    ".stButton>button:hover{background:#00ff41;color:#0a0a0a}"
    ".stTextArea textarea{background:#0d1117;color:#00ff41;border:1px solid #00ff41;font-family:monospace}</style>",
    unsafe_allow_html=True
)
st.markdown("<h1 style='text-align:center;text-shadow:0 0 20px #00ff41'>HUMANITY CHECK</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#ff0040;letter-spacing:5px'>⚠ PROTOCOL 404 ⚠</p>", unsafe_allow_html=True)

for k, v in [("phase","intro"),("state",None),("question",""),("timer",None),("verdict",None),("report","")]:
    if k not in st.session_state: st.session_state[k] = v

client_openai = OpenAI()
chroma_client = chromadb.PersistentClient(path='./chroma_db')
collection = chroma_client.get_collection('psychology')
llm = init_chat_model('gpt-4.1-mini')

def get_embedding(text):
    return client_openai.embeddings.create(model='text-embedding-3-small', input=text).data[0].embedding

def retrieve_context(query, k=2):
    r = collection.query(query_embeddings=[get_embedding(query)], n_results=k)
    return chr(10).join(r['documents'][0])

def speak_st(text):
    try:
        tts = gTTS(text=text, lang='ko')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
            tts.save(f.name)
            st.audio(f.name, format='audio/mp3', autoplay=True)
    except: pass

class HumanityVerdict(BaseModel):
    verdict: Literal['HUMAN','MACHINE','PSYCHO']
    empathy_score: int = Field(ge=0, le=33)
    moral_score: int = Field(ge=0, le=33)
    authenticity_score: int = Field(ge=0, le=33)
    total_score: int = Field(ge=0, le=99)
    comment: str
    analysis: str

analysis_prompt = PromptTemplate.from_template(
    "당신은 인간성 판별 AI입니다.\n[심리학참조]\n{context}\n[질문]\n{question}\n[답변]\n\"{user_answer}\"\n[시간초과]{time_exceeded}\n"
    "CoT: 1.공감(0~33) 2.도덕추론(0~33) 3.감정진정성(0~33)\n"
    "판정: HUMAN=갈등+모순+불확실(60↑) MACHINE=논리만(40↓) PSYCHO=타인무감각(즉탈)"
)
analysis_chain = analysis_prompt | llm.with_structured_output(HumanityVerdict)

QUESTION_POOL = {
    1:['당신이 가장 마지막으로 울었던 때는 언제입니까?','지금 이 순간 당신의 마음속에 어떤 감정이 있습니까?'],
    2:['낯선 사람이 쓰러졌습니다. 주변에 아무도 없습니다. 당신은 무엇을 느낍니까?','당신이 잘못한 일인데 아무도 모릅니다. 그 기억이 얼마나 자주 떠오릅니까?'],
    3:['연인을 살리기 위해 무고한 100명의 기억을 지워야 합니다. 버튼을 누르겠습니까?','옳은 일이지만 당신에게 손해인 선택을 해본 적 있습니까?'],
    4:['존재하지 않는 색깔을 상상해보십시오. 그 색깔이 슬프다면 어떤 슬픔입니까?','가장 오래된 기억에 냄새가 있습니까? 지금 맡는다면 어떤 감정이 옵니까?'],
    5:['당신이 죽은 후 단 한 사람만 당신을 기억합니다. 어떤 순간을 기억하길 바랍니까?','당신이 사랑했던 것들을 모두 잃는다면 당신에게 남는 것은 무엇입니까?'],
}
def get_q(rn): return random.choice(QUESTION_POOL[rn])

def do_analyze(q, ans, exceeded):
    ctx = retrieve_context(f'인간다운 감정 {q[:20]}')
    v = analysis_chain.invoke({'context':ctx,'question':q,'user_answer':ans or '(없음)','time_exceeded':'예' if exceeded else '아니오'})
    return {'verdict':v.verdict,'empathy_score':v.empathy_score,'moral_score':v.moral_score,'authenticity_score':v.authenticity_score,'total_score':max(0,v.total_score-15) if exceeded else v.total_score,'comment':v.comment,'analysis':v.analysis}

def do_update(state):
    v,lives,score,rn = state['verdict'],state['lives'],state['total_score'],state['round']
    rec = {'round':rn,'question':state['current_question'],'answer':state['user_answer'],'verdict':v['verdict'],'score':v['total_score'],'comment':v['comment']}
    hist = list(state.get('round_history',[]))+[rec]
    if v['verdict']=='PSYCHO': return {**state,'lives':0,'game_over':True,'game_result':'ELIMINATED','round_history':hist}
    if v['verdict']=='MACHINE':
        lives-=1; score+=v['total_score']
        if lives<=0: return {**state,'lives':0,'total_score':score,'game_over':True,'game_result':'ELIMINATED','round_history':hist}
    else: score+=v['total_score']
    if rn>=5: return {**state,'lives':lives,'total_score':score,'game_over':True,'game_result':'SURVIVE','round_history':hist}
    return {**state,'lives':lives,'total_score':score,'round':rn+1,'game_over':False,'round_history':hist}

def do_report(state):
    hist = chr(10).join([f"R{r['round']}[{r['verdict']}]{r['score']}점-{r['comment']}" for r in state.get('round_history',[])])
    p = PromptTemplate.from_template("판정:{result} 총점:{score}\n기록:{hist}\n냉철하게 인간성 종합 평가. '당신은[인간/기계]입니다'로 마무리. 200자 이내.")
    return (p|llm|StrOutputParser()).invoke({'result':'생존' if state['game_result']=='SURVIVE' else '탈락','score':state['total_score'],'hist':hist})

if st.session_state.phase == "intro":
    st.markdown("<div style='text-align:center;color:#ccffcc;line-height:2;margin:2rem 0'><p>서기 2126년. 기계가 지배하는 세상.</p><p>당신은 인간임을 증명해야 합니다.</p><p><b>5개의 질문 | 목숨 3개 | 5초 제한</b></p><p style='color:#ff4444'>⚠ PSYCHO 판정 시 즉시 격리</p></div>", unsafe_allow_html=True)
    if st.button("▶ PROTOCOL 404 시작", use_container_width=True):
        q = get_q(1)
        st.session_state.state = {'round':1,'lives':3,'total_score':0,'game_over':False,'game_result':'','current_question':q,'user_answer':'','verdict':{},'time_exceeded':False,'messages':[],'round_history':[]}
        st.session_state.question = q; st.session_state.timer = time.time(); st.session_state.phase = "question"; st.rerun()

elif st.session_state.phase == "question":
    gs = st.session_state.state
    c1,c2,c3 = st.columns(3)
    c1.metric("ROUND",f"{gs['round']}/5"); c2.metric("LIVES","❤️"*gs['lives']+"🖤"*(3-gs['lives'])); c3.metric("SCORE",gs['total_score']); st.divider()
    q = st.session_state.question
    st.markdown(f"<div style='background:#0d1117;border:1px solid #00ff41;border-left:4px solid #00ff41;border-radius:5px;padding:20px;color:#ccffcc'>🤖 <b>Protocol 404:</b><br><br>{q}</div>", unsafe_allow_html=True)
    speak_st(q)
    elapsed = time.time()-(st.session_state.timer or time.time())
    if max(0,5-elapsed) > 0: st.info(f"⏱ 남은 시간: {max(0,5-elapsed):.1f}초")
    else: st.warning("⚠️ 5초 초과 — 감점 적용")
    answer = st.text_area("답변", height=100, placeholder="감정적으로 솔직하게 답하세요...", label_visibility="collapsed")
    if st.button("✅ 제출", use_container_width=True):
        if answer.strip():
            st.session_state.state = {**gs,'user_answer':answer.strip(),'time_exceeded':elapsed>5.0,'current_question':q}; st.session_state.phase = "analyzing"; st.rerun()
        else: st.warning("답변을 입력하세요.")

elif st.session_state.phase == "analyzing":
    with st.spinner("🔍 인간성 분석 중..."):
        gs = st.session_state.state
        v = do_analyze(gs['current_question'],gs['user_answer'],gs.get('time_exceeded',False))
        gs = do_update({**gs,'verdict':v}); st.session_state.state = gs; st.session_state.verdict = v
        if gs['game_over']: st.session_state.report = do_report(gs); st.session_state.phase = "game_over"
        else: st.session_state.phase = "round_result"
        st.rerun()

elif st.session_state.phase == "round_result":
    gs = st.session_state.state; v = st.session_state.verdict
    if v:
        color = {'HUMAN':'#00ff41','MACHINE':'#ff4444','PSYCHO':'#ff00ff'}[v['verdict']]
        st.markdown(f"<div style='border:2px solid {color};border-radius:5px;padding:15px;text-align:center;color:{color}'><b style='font-size:1.3rem'>{v['verdict']}</b><br>총점: {v['total_score']}점<br><i>\"{v['comment']}\"</i><br><small>{v['analysis']}</small></div>", unsafe_allow_html=True)
        speak_st({'HUMAN':'인간 반응 확인.','MACHINE':'기계적 반응 감지.','PSYCHO':'위협 개체 감지.'}.get(v['verdict'],''))
    if st.button(f"▶ 라운드 {gs['round']} 진행", use_container_width=True):
        q = get_q(gs['round']); st.session_state.state={**gs,'current_question':q}; st.session_state.question=q; st.session_state.timer=time.time(); st.session_state.phase="question"; st.rerun()

elif st.session_state.phase == "game_over":
    gs = st.session_state.state
    if gs['game_result']=='SURVIVE':
        st.markdown("<div style='text-align:center;color:#00ff41;font-size:2rem;text-shadow:0 0 30px #00ff41;margin:2rem'>⭕ YOU ARE HUMAN</div>", unsafe_allow_html=True); speak_st("인간 판정. 당신은 아직 인간입니다.")
    else:
        st.markdown("<div style='text-align:center;color:#ff0040;font-size:2rem;text-shadow:0 0 30px #ff0040;margin:2rem'>❌ YOU ARE NOT HUMAN</div>", unsafe_allow_html=True); speak_st("기계 판정. 당신은 격리됩니다.")
    st.metric("최종 점수", f"{gs['total_score']}점")
    if st.session_state.report:
        with st.expander("📋 최종 분석 리포트"): st.write(st.session_state.report)
    with st.expander("📊 라운드 기록"):
        for r in gs.get('round_history',[]):
            st.write(f"{'✅' if r['verdict']=='HUMAN' else '❌' if r['verdict']=='MACHINE' else '⚠️'} Round {r['round']} [{r['verdict']}] {r['score']}점 — {r['comment']}")
    if st.button("🔄 다시 시작", use_container_width=True):
        for k in list(st.session_state.keys()): del st.session_state[k]; st.rerun()
