from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder,ChatPromptTemplate
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.runnables import RunnableLambda,RunnableParallel
from operator import itemgetter
from qdrant_client import QdrantClient
import os

load_dotenv()
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
COLLECTION = os.getenv("QDRANT_COLLECTION", "test")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


parser=StrOutputParser()
model=ChatOpenAI()
embed=OpenAIEmbeddings(model="text-embedding-3-small")

#loader=PyPDFLoader('Final_SHB_Spring_2025.pdf')
PDF_PATH = os.environ.get("PDF_PATH", "")
if not PDF_PATH:
    raise ValueError("PDF_PATH not set")

loader = PyPDFLoader(PDF_PATH)

doc = list(loader.lazy_load())
split = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
chunk = split.split_documents(doc)
def get_vectorstore():
    try:
        client.get_collection(COLLECTION)

        return QdrantVectorStore(
            client=client,
            collection_name=COLLECTION,
            embedding=embed,
        )
    except Exception:

        return QdrantVectorStore.from_documents(
            documents=chunk,
            embedding=embed,
            url=QDRANT_URL,       
            api_key=QDRANT_API_KEY,
            collection_name=COLLECTION,
        )

vectorstore = get_vectorstore()

retriever=vectorstore.as_retriever(search_kwargs={'k':4},search_type='mmr')


prompt = ChatPromptTemplate.from_messages([
    ("system","""
    You are a helpful assistant.
    Answer only from the provided context in max 2 lines. If the context is insufficient, just say "I don't know."
    """),
    MessagesPlaceholder("history"),
    ("human", "Question: {question}\n\nContext:\n{text}")
])

def formatted_doc(retrieved_doc):
    context = "\n\n".join(d.page_content for d in retrieved_doc)
    return context


parallel_chain=RunnableParallel({
    "text": itemgetter("question")|retriever|RunnableLambda(formatted_doc),
    "question": itemgetter("question"),
    "history": itemgetter("history")})

chain=parallel_chain|prompt|model|parser

history=[]

while True:
    query=input("Ask me the question regrading pdf: ")
    if query.lower() in {"exit", "quit", "q"}:
        print("Goodbye!")
        break

    res=chain.invoke({
         "history": history,
        "question": query
    })

    print("\nAnswer:", res, "\n")
    history.append(HumanMessage(content=query))
    history.append(AIMessage(content=res))

