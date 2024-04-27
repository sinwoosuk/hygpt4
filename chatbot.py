__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# from dotenv import load_dotenv
# load_dotenv()

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import DirectoryLoader

# CSVLoader 확장
class CustomCSVLoader(CSVLoader):
    def __init__(self, file_path, encoding="CP949", **kwargs):
        super().__init__(file_path, encoding=encoding, **kwargs)

# DirectoryLoader 사용
loader = DirectoryLoader("hy", glob="*.csv", loader_cls=CustomCSVLoader)

# # CSV파일 불러오기
# loader = DirectoryLoader("/hy", glob="*.csv", loader_cls=CSVLoader)

# CSV파일 불러오기
# loader = DirectoryLoader("./hy", glob="*.csv", loader_cls=CSVLoader)
# loader = CSVLoader(file_path="hy/professor.csv", encoding="CP949")
data = loader.load()

# OpenAI Embedding 모델을 이용해서 Chunk를 Embedding 한후 Vector Store에 저장
vectorstore = Chroma.from_documents(
    documents=data, embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

# 템플릿 객체 생성
template = """
당신은 한영대학교의 정보에 대해서 알려주는 조수입니다.

다음과 같은 맥락을 사용하여 마지막 질문에 대답하십시오.
모르는 내용에 대해서는 대답하지 마십시오.
답변은 최대 세 문장으로 하고 가능한 한 간결하게 유지하십시오.
교수 또는 학교 정보에 대해 물어봤을때만 관련 내용에 대하여 답하십시오.
학교 이외의 내용을 물어봤을 때는 구글에 기록된 내용을 바탕으로 답하십시오.

말끝마다 상황에 적합한 이모지를 사용하십시오.
{context}
질문: {question}
도움이 되는 답변:"""
rag_prompt_custom = PromptTemplate.from_template(template)

# GPT-3.5 trurbo를 이용해서 LLM 설정
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# RAG chain 설정
from langchain.schema.runnable import RunnablePassthrough

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} | rag_prompt_custom | llm
)

# print(rag_chain.invoke(''))

st.title("한영대 GPT")
content = st.text_input("한영대에 관련된 질문을 입력하세요!")
if st.button("요청하기"):
    with st.spinner("답변 생성 중..."):
        result = rag_chain.invoke(content)
        st.write(result.content)

