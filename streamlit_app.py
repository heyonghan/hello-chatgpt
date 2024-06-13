import streamlit as st
from langchain.memory import VectorStoreRetrieverMemory
from langchain_openai import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_zhipu import ZhipuAIEmbeddings

import os


# Set your OpenAI API key as an environment variable
os.environ["ZHIPUAI_API_KEY"] = "d040453f81537fd7c1b3687ec810f25f.BTT6DTB5CVlS7sCf"

def create_conversation_chain():
    vectorstore = Chroma(embedding_function=ZhipuAIEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
    memory = VectorStoreRetrieverMemory(retriever=retriever)
    memory.save_context({"input": "我喜欢学习"}, {"output": "你真棒"})
    memory.save_context({"input": "我不喜欢玩儿"}, {"output": "你可太棒了"})
    PROMPT_TEMPLATE = """以下是人类和 AI 之间的友好对话。AI 话语多且提供了许多来自其上下文的具体细节。如果 AI 不知道问题的答案，它会诚实地说不知道。
以前对话的相关片段：
{history}
（如果不相关，你不需要使用这些信息）
当前对话：
人类：{input}
AI：
"""
    prompt = PromptTemplate(input_variables=["history", "input"], template=PROMPT_TEMPLATE)
    chat_model = ChatZhipuAI(
        model="glm-4",
        temperature=0.5,
    )
    conversation_with_summary = ConversationChain(
        llm=chat_model,
        prompt=prompt,
        memory=memory,
        verbose=True
    )
    return conversation_with_summary

st.title("对话应用")

if 'conversation_chain' not in st.session_state:
    st.session_state['conversation_chain'] = create_conversation_chain()

user_input = st.text_input("你说：")
if user_input:
    response = st.session_state['conversation_chain'].predict(input=user_input)
    st.text_area("AI 回复：", value=response)

