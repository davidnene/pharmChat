from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient
# from ingest import embeddings, url

import txtai

embeddings = txtai.Embeddings(path="neuml/pubmedbert-base-embeddings")

# embeddings = SentenceTransformerEmbeddings(model_name = "NeuML/pubmedbert-base-embeddings")

url = "http://localhost:6333/dashboard"

client = QdrantClient(
    url=url,
    prefer_grpc=False
)

print(client)

db = Qdrant(client=client, embeddings=embeddings, collection_name='vector_database')
print(db)
print("####################")
query = "what is Halothane?"

docs = db.similarity_search_with_score(query=query, k=3)

for i in docs:
    doc, score = i
    print({"score": score, "Content":doc.page_content, "metadata":doc.metadata})