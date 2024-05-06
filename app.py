# -*- coding: utf-8 -*-
# @Author: Richard
import logging
import os
import time

import openai
import streamlit as st
from dotenv import load_dotenv
from langchain import FAISS
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
from pypdf import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space

# 设置OpenAI的API密钥和 URL
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE', "https://openrouter.ai/api/v1")
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
print(f"os.environ: {os.environ} \n")

openai.api_key = OPENAI_API_KEY
openai.api_base = OPENAI_API_BASE


class GetHuggingfaceEmbeddingsException(Exception):
    pass


class GetHuggingfaceLLMException(Exception):
    pass


def get_pdf_text(pdf_reader, pages=None):
    """
    提取PDF文本并按页存储
    :param pdf_reader:
    :param pages:
    :return:
    """
    text_dict = {}
    text = ""
    if pages:
        for i, page in enumerate(pdf_reader.pages):
            if i + 1 in pages:
                text += f"#### ----- Page {i + 1} ----- :\n\n" + page.extract_text()
                text_dict[i + 1] = page.extract_text()
    else:
        for i, page in enumerate(pdf_reader.pages):
            text += f"#### ----- Page {i + 1} ----- :\n\n" + page.extract_text()
            text_dict[i + 1] = page.extract_text()
    return text, text_dict


def get_chunks_dict(text_dict):
    """
    将文本分割成小块
    :param text_dict:
    :return:
    """
    chunks_dict = {}
    for page, content in text_dict.items():
        # 设置 overlap 单词重叠来保证数据不丢失
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=4000,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_text(content)
        chunks_dict[page] = chunks
    return chunks_dict


def get_full_chunks(chunks_dict):
    """
    合并所有文本块
    :param chunks_dict:
    :return:
    """
    full_chunks = []
    for page, chunks in chunks_dict.items():
        for chunk in chunks:
            full_chunks.append(chunk)
    return full_chunks


def answer(question):
    # 检查会话状态中是否存在文档，如果不存在，则从知识库中搜索
    if "docs" not in st.session_state:
        # 执行相似性搜索
        st.session_state.docs = st.session_state.knowledge_base.similarity_search(query=question)
    st.session_state.page_content = st.session_state.docs[0].page_content
    # 运行链式处理并获取响应
    response = st.session_state.chain.run(question=question, docs=st.session_state.page_content)
    response = response.replace("\n", "")
    text = """the answer is ALWAYS "I don't know!" No other answer is allowed."""
    response = response.split(text)[1]
    return response


def get_huggingface_embeddings():
    """ 使用 HuggingFaceEmbeddings 生成文本块的嵌入向量，并用FAISS创建向量存储 """
    model_name = "maidalun1020/bce-embedding-base_v1"
    model_kwargs = {"device": "cpu"}
    # 设置 embedding 文本大小为 64, 标准化 embedding
    encode_kwargs = {"batch_size": 64, "normalize_embeddings": True}
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
    except Exception as e:
        logging.error(e)
        raise GetHuggingfaceEmbeddingsException
    return embeddings


def get_huggingface_llm():
    """ 获取 huggingface llm """
    try:
        llm = HuggingFaceHub(
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
            repo_id='mistralai/Mistral-7B-Instruct-v0.2',
        )
    except Exception as e:
        logging.error(e)
        raise GetHuggingfaceLLMException
    return llm


def main():
    # 加载环境变量
    load_dotenv()
    # 设置应用程序标题和图标
    st.set_page_config(page_title="Chat with PDF", page_icon=":books:", layout="wide")
    st.header("Chat with PDF :books:")

    # 创建侧边栏以帮助用户和上传文档
    with st.sidebar:
        help_message = """
        欢迎使用 Chat PDF 应用

        1. 请在侧边栏上传您的 PDF 文件。
        2. 点击 Process 按钮，从 PDF 中提取文本。
        3. 您可以通过聊天界面提问与聊天机器人进行互动。
        4. 如需重置聊天记录，请点击 Clear Conversation 按钮。
        5. 要关闭本帮助信息，请点击 Close 按钮。

        祝您聊天愉快！
        """

        if st.button("Help!"):
            with st.expander("Help"):
                st.info(help_message)

        # 定义PDF上传器
        pdf_docs = st.file_uploader("在此上传您的 PDF，然后点击 'Process' 生成")

        if st.button("Process"):
            if pdf_docs:
                st.session_state.pdf_reader = PdfReader(pdf_docs)
                with st.spinner("Processing.This may take a while⏳"):
                    st.session_state.full_text, st.session_state.text_dict = get_pdf_text(st.session_state.pdf_reader)
                    st.session_state.chunks_dict = get_chunks_dict(st.session_state.text_dict)
                    st.session_state.full_chunks = get_full_chunks(st.session_state.chunks_dict)
                    st.write("Processing done!")

                    st.session_state.knowledge_base = None
            else:
                st.write("Please upload your document!")

        if "full_text" in st.session_state:
            st.write("内容摘要:")
            summary_prompt_template = """
            Create an accurate, clear and useful PDF summary based on the following 
            to help readers quickly understand the core content of the document:
            """
            full_text = ''
            for page, content in st.session_state.text_dict.items():
                full_text += content
            # 生成内容摘要
            full_text = summary_prompt_template + full_text
            summary_result = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[dict(role="user", content=full_text)]
            )

            response = summary_result.choices[0].message.content
            st.write(response)
            # 提示词的 prompt
            prompt = '通过阅读下面的文档，生成可能涉及到的问题，请确保相关性：' + full_text
            summary_result = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[dict(role="user", content=prompt)]
            )

            response = summary_result.choices[0].message.content
            st.write(response)

        add_vertical_space()

    # 创建聊天区域
    col1, _ = st.columns([1, 5])

    # 重置按钮
    with col1:
        reset_chat = st.button("Clear Conversation")

    # 定义系统提示模板
    st.session_state.system_prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
            You are a helpful assistant that can answer question about the uploaded document.
            Answer the following question: {question}
            By searching the following document: {docs}
            Only use the factual information from the document to answer the question.
            Your answer should be detailed.
            If you don't have enough information to answer the question, 
            the answer is ALWAYS "I don't know!" No other answer is allowed.
        """)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if reset_chat:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if "full_text" in st.session_state:
        if prompt := st.chat_input("Ask any question about the document?"):
            start_time = time.time()
            if "embeddings" not in st.session_state:
                st.session_state.embeddings = get_huggingface_embeddings()

            if st.session_state.knowledge_base is None:
                # 使用 FAISS 向量数据库存储 embeddings, 构建一个基于文本数据的知识库
                st.session_state.knowledge_base = FAISS.from_texts(
                    st.session_state.full_chunks,
                    st.session_state.embeddings
                )

            if "llm" not in st.session_state:
                st.session_state.llm = get_huggingface_llm()

            if "chain" not in st.session_state:
                st.session_state.chain = LLMChain(
                    llm=st.session_state.llm,
                    prompt=st.session_state.system_prompt
                )

            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()

                full_response = answer(prompt)
                print(full_response)
                message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            end_time = time.time()
            elapsed_time = end_time - start_time
            time_span = f'<span style="font-size: 0.75rem;">处理耗时：{elapsed_time:.2f}秒</span>'
            # 显示耗时信息
            st.markdown(time_span, unsafe_allow_html=True)

    else:
        st.write("Please upload your document and press Process before asking any questions.")


if __name__ == "__main__":
    main()
