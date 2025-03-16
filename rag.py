from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Qdrant
import os
import json
from huggingface_hub import login

# Set your Hugging Face Token
HF_TOKEN = "************"  # Replace with your actual token

# Login to Hugging Face Hub
login(HF_TOKEN)

# FastAPI Initialization
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Hugging Face model name (replace with actual model repo)
MODEL_NAME = "TheBloke/Meditron-7B-GGUF"  # Replace with the actual model name

# Configuration for CTransformers
config = {
    'max_new_tokens': 1024,
    'context_length': 2048,
    'repetition_penalty': 1.1,
    'temperature': 0.1,
    'top_k': 50,
    'top_p': 0.9,
    'threads': int(os.cpu_count() / 2)  # Adjusting thread count
}

# Load the LLM model directly from Hugging Face
llm = CTransformers(
    model=MODEL_NAME,
    model_type="llama",
    lib="avx2",
    hf=True,  # Enables downloading from Hugging Face
    hf_auth=HF_TOKEN,  # Authenticates with Hugging Face
    **config
)

print("LLM Initialized from Hugging Face...")

# Prompt Template
prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

# Initialize Embeddings
embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

# Qdrant Connection
QDRANT_URL = "http://localhost:6333"
client = QdrantClient(url=QDRANT_URL, prefer_grpc=False)

# Load the vector database
db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db")

# Create retriever
retriever = db.as_retriever(search_kwargs={"k": 1})

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get_response")
async def get_response(query: str = Form(...)):
    # Define the QA Chain
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True
    )

    # Get response
    response = qa(query)
    print(response)

    # Extract response details
    answer = response['result']
    source_document = response['source_documents'][0].page_content
    doc = response['source_documents'][0].metadata.get('source', 'Unknown')

    # Convert to JSON response
    response_data = jsonable_encoder({
        "answer": answer,
        "source_document": source_document,
        "doc": doc
    })

    return Response(json.dumps(response_data), media_type="application/json")
