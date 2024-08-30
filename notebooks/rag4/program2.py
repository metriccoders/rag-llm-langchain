from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
import os
os.environ["OPENAI_API_KEY"] = ""
Settings.llm = OpenAI(temperature=0.2, model="gpt-4")

data_path = "../../datasets/data"
persist_dir = "./storage"
if not os.path.exists(persist_dir):
    documents = SimpleDirectoryReader(data_path).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=persist_dir)
else:
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context=storage_context)

query_engine = index.as_query_engine()

while True:
    question = input("Ask your question: ")
    response = query_engine.query(question)
    print(response)
