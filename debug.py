import streamlit as st
from LLMmodel import LLM_model
from chromadb import PersistentClient
from retrieve_model import retrieve_relevant_chunks

st.set_page_config(page_title="RAG问答工具", layout="centered")
st.title("RAG 问答工具")

# 初始化
if "initialized" not in st.session_state:
    st.session_state.client = PersistentClient(path="./chroma_db")
    st.session_state.collection = st.session_state.client.get_collection(name="document_embeddings")
    st.session_state.llm = LLM_model()
    with st.spinner("启动LLM模型中..."):
        st.session_state.llm.start_LLM()
    st.session_state.initialized = True
    st.success("初始化完成，可以开始提问")

query = st.text_input("请输入你的问题：", key="user_query")

if st.button("提交查询") and query:
    with st.spinner("正在检索相关文档..."):
        chunks = retrieve_relevant_chunks(user_query=query,
                                          collection=st.session_state.collection)

    # 流式输出
    answer_container = st.empty()           # 动态容器
    answer = ""
    st.subheader("回答结果：")
    for delta in st.session_state.llm.call_llm_stream(query, chunks):
        answer += delta
        answer_container.write(answer)

    # 折叠展示检索片段
    with st.expander("查看检索到的相关片段", expanded=False):
        for i, chunk in enumerate(chunks, 1):
            st.text_area(f"片段 {i}", chunk, disabled=True, height=100)