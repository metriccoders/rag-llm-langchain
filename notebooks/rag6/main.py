from llama_index.core import VectorStoreIndex, get_response_synthesizer, SimpleDirectoryReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

import os
os.environ["OPENAI_API_KEY"] = ""

data_path = "../../datasets/data"

documents = SimpleDirectoryReader(data_path).load_data()

index = VectorStoreIndex.from_documents(documents=documents)

retriever = VectorIndexRetriever(index=index, similarity_top_k=10)


response_synthesizer = get_response_synthesizer()

query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer, node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],)


while True:
    question = input("Ask a question:")
    response = query_engine.query(question)
    print(response)