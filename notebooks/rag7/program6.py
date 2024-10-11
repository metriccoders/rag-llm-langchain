from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

documents = SimpleDirectoryReader("data").load_data()

Settings.embed_model = OllamaEmbedding(model_name="llama3.2:1b")

Settings.llm = Ollama(model="llama3.2:1b", request_timeout=360.0)

index=VectorStoreIndex.from_documents(documents=documents)

query_engine = index.as_query_engine()

response = query_engine.query("What were the main market challenges in 2021 in the document?")

print(response)