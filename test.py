import streamlit as st
from RAGlibrary import RAG

st.set_page_config(page_title="RAG问答工具", layout="centered")
st.title("RAG 问答工具")

# 初始化RAG
if "rag" not in st.session_state:
    with st.spinner("初始化RAG系统中..."):
        st.session_state.rag = RAG()
    st.success("初始化完成，可以开始提问")

query = st.text_input("请输入你的问题：", key="user_query")

if st.button("提交查询") and query:
    # 流式输出回答
    answer_container = st.empty()
    answer = ""
    st.subheader("回答结果：")
    for delta in st.session_state.rag.call_RAG(query):
        answer += delta
        answer_container.write(answer)
