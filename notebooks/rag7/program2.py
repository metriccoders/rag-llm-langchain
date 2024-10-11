from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage

import os
os.environ["OPENAI_API_KEY"] = ""


documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents=documents)

query_engine = index.as_query_engine()
response = query_engine.query("What was the revenue in 2021?")
print(response)