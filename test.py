import streamlit as st
import os
from pathlib import Path
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda, RunnableParallel
from operator import itemgetter
from qdrant_client import QdrantClient

load_dotenv()
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
#COLLECTION = os.getenv("QDRANT_COLLECTION", "test")

parser = StrOutputParser()
model = ChatOpenAI()
embed = OpenAIEmbeddings(model="text-embedding-3-small")

st.title("RAG Based Application")

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded:
    os.makedirs("uploads", exist_ok=True)
    safe_name = Path(uploaded.name).name
    save_path = os.path.join("uploads", safe_name)
    with open(save_path, "wb") as f:
        f.write(uploaded.read())

    st.success(f"Uploaded: {safe_name}")
    loader = PyPDFLoader(save_path)
    docs = list(loader.lazy_load())
    split = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = split.split_documents(docs)
    
    name=Path(safe_name).stem
    COLLECTION=name
    
    
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    try:
        client.get_collection(COLLECTION)
        vectorstore = QdrantVectorStore(client=client, collection_name=COLLECTION, embedding=embed)
    except Exception:
        vectorstore = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embed,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name=COLLECTION,
        )

    retriever = vectorstore.as_retriever(search_kwargs={'k': 4}, search_type='mmr')

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a helpful assistant.
        Answer only from the provided context in max 2 lines. 
        If the context is insufficient, just say "I don't know."
        """),
        MessagesPlaceholder("history"),
        ("human", "Question: {question}\n\nContext:\n{text}")
    ])

    def formatted_doc(retrieved_doc):
        context = "\n\n".join(d.page_content for d in retrieved_doc)
        return context

    parallel_chain = RunnableParallel({
        "text": itemgetter("question") | retriever | RunnableLambda(formatted_doc),
        "question": itemgetter("question"),
        "history": itemgetter("history")
    })

    chain = parallel_chain | prompt | model | parser
    history = []

    question = st.text_input("Ask a question about the PDF:")
    if question:
        res = chain.invoke({
            "history": history,
            "question": question
        })
        st.write("**Answer:**", res)
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=res))
