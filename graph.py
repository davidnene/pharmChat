import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from neo4j import GraphDatabase
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI


load_dotenv()
graph = Neo4jGraph()
loader = DirectoryLoader('Data/', glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)

documents = loader.load()
print('Documents loaded successfully...')

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100
)

texts = text_splitter.split_documents(documents)
print('Documents splitted successfully...')
llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")
llm_transformer = LLMGraphTransformer(llm=llm)

print('LLM graph transformer loaded successfully...')

print('converting to graph documents..')
graph_documents = llm_transformer.convert_to_graph_documents(texts)

print('converting to graph documents successfull..')
print('Data ingestion to Graph db..')
graph.add_graph_documents(
    graph_documents,
    baseEntityLabel=True,
    include_source=True
)

print('Data ingestion to Graph db successfull..')