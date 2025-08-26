from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import MongodbLoader
from bs4 import BeautifulSoup
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.chains.question_answering import load_qa_chain
import os
from dotenv import load_dotenv
import asyncio



# Load API keys from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
MONGO_URI = "mongodb://localhost:27017/"

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev, allow all. Later, restrict to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === 1. HTML to plain text ===
def html_to_plain_text(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for tag in soup(['script', 'style', 'img']):
        tag.decompose()
    return soup.get_text(separator=' ', strip=True)

# === 2. Load MongoDB articles (ASYNC) ===
async def load_articles():
    loader = MongodbLoader(
        connection_string=MONGO_URI,
        db_name="dummy",
        collection_name="items",
        field_names=["content"]
    )
    # Use async load
    documents = await loader.aload()

    cleaned_docs = []
    for doc in documents:
        plain_text = html_to_plain_text(doc.page_content)
        cleaned_docs.append(plain_text)

    return "\n\n---\n\n".join(cleaned_docs)

# === 3. Setup vectorstore ===
async def setup_vectorstore():
    articles = await load_articles()

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)
    text_splitter = SemanticChunker(embedding_model)
    docs = text_splitter.create_documents([articles])

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "blogguru-index"
    dimension = 3072

    if index_name not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    return PineconeVectorStore.from_documents(
        docs, embedding=embedding_model, index_name=index_name
    )
    

# Global placeholders
vector_index = None
chain = None

# === 4. Initialize on startup ===
@app.on_event("startup")
async def startup_event():
    global vector_index, chain
    vector_index = await setup_vectorstore()
    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type='stuff')

# === 5. Query handling ===
def retrieve_answers(query, k=2):
    matching_res = vector_index.similarity_search(query, k=k)
    return chain.run(input_documents=matching_res, question=query)

# === 6. API endpoint ===
class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        answer = retrieve_answers(req.query)
        return {"response": answer}
    except Exception as e:
        return {"error": str(e)}
