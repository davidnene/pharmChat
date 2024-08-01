from langchain_core.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant
from langchain_community.vectorstores import Qdrant
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model directly
load_dotenv()
token = os.getenv("HUGGINGFACE_API_TOKEN")
headers = {"Authorization": f"Bearer {token}"}
tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-7b", token=token)
model = AutoModelForCausalLM.from_pretrained("epfl-llm/meditron-7b", token=token)

# tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-7b")
# model = AutoModelForCausalLM.from_pretrained("epfl-llm/meditron-7b")
local_llm_2 = "meditron-7b.Q5_K_M.gguf"
local_llm = model

config = {
'max_new_tokens': 1024,
'context_length': 2048,
'repetition_penalty': 1.1,
'temperature': 0.1,
'top_k': 50,
'top_p': 0.9,
'stream': True,
'threads': int(os.cpu_count() / 2)
}

llm = CTransformers(
    model=local_llm_2,
    model_type="llama",
    lib="avx2",
    api_key=token,
    **config
)

print("LLM Initialized....")

prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")


url = "http://localhost:6333"

client = QdrantClient(
    url=url, prefer_grpc=False
)

db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_database")

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

retriever = db.as_retriever(search_kwargs={"k":1})
chain_type_kwargs = {"prompt": prompt}
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs, verbose=True)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get_response")
async def get_response(query: str = Form(...)):
    response = qa(query)
    print(response)
    answer = response['result']
    source_document = response['source_documents'][0].page_content
    doc = response['source_documents'][0].metadata['source']
    response_data = jsonable_encoder(json.dumps({"answer": answer, "source_document": source_document, "doc": doc}))
    
    res = Response(response_data)
    print(res)
    return res