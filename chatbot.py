import sys
# from dotenv import load_dotenv
# load_dotenv()
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import DirectoryLoader
from PIL import Image
import os

# CSVLoader 확장
class CustomCSVLoader(CSVLoader):
    def __init__(self, file_path, encoding="CP949", **kwargs):
        super().__init__(file_path, encoding=encoding, **kwargs)

# DirectoryLoader 사용
loader = DirectoryLoader("hy", glob="*.csv", loader_cls=CustomCSVLoader)
# CSV파일 불러오기
data = loader.load()
# OpenAI Embedding 모델을 이용해서 Chunk를 Embedding 한후 Vector Store에 저장
vectorstore = Chroma.from_documents(
    documents=data, embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()
# 템플릿 객체 생성
template = """
다음과 같은 맥락을 사용하여 마지막 질문에 대답하십시오.
답변은 최대 세 문장으로 하고 가능한 한 간결하게 유지하십시오.
답변은 구글 검색을 기반으로 대답하십시오.
{context}
질문: {question}
도움이 되는 답변:"""
rag_prompt_custom = PromptTemplate.from_template(template)
# GPT-3.5 trurbo를 이용해서 LLM 설정
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-4", temperature=0)
# RAG chain 설정
from langchain.schema.runnable import RunnablePassthrough
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} | rag_prompt_custom | llm
)

# 이미지 파일 경로 설정
logo_img_path = os.path.join(os.getcwd(), 'logo.png')

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://i.imgur.com/MVw5Fuw.png");
             background-attachment: fixed;
             background-size: cover
             
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 

# 로고 이미지 삽입
# img = Image.open(logo_img_path)
# st.image(img)
st.title("한영대 GPT")
content = st.text_input("한영대에 관련된 질문을 입력하세요!")
if st.button("요청하기"):
    with st.spinner("답변 생성 중..."):
        result = rag_chain.invoke(content)
        st.write(result.content)
