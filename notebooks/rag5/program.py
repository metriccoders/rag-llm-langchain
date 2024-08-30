import chromadb
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import os
os.environ["OPENAI_API_KEY"] = ""
Settings.llm = OpenAI(temperature=0.2, model="gpt-4")

data_path = "../../datasets/data"
persist_dir = "./storage"

documents = SimpleDirectoryReader(data_path).load_data()
db = chromadb.PersistentClient(path="./chroma_db")

chroma_collection = db.get_or_create_collection("quickstart")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)


query_engine = index.as_query_engine()

while True:
    question = input("Ask your question: ")
    response = query_engine.query(question)
    print(response)


