import os.path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
import os
os.environ["OPENAI_API_KEY"] = ""


# check if storage already exists
PERSIST_DIR = "./storage"

if not os.path.exists(PERSIST_DIR):
    documents = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(documents=documents)
    
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context=storage_context)
    

query_engine = index.as_query_engine()

question="""
    Net Profit for 2021
"""

response = query_engine.query(question)
print(response)