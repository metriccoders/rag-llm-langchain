from llama_index.core import Settings, VectorStoreIndex, get_response_synthesizer, SimpleDirectoryReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor, KeywordNodePostprocessor
from llama_index.llms.openai import OpenAI

import os
os.environ["OPENAI_API_KEY"] = ""
Settings.llm = OpenAI(temperature=0.2, model="gpt-4")

data_path = "../../datasets/data"

documents = SimpleDirectoryReader(data_path).load_data()

index = VectorStoreIndex.from_documents(documents)

retriever = VectorIndexRetriever(
    index=index, similarity_top_k=10
)

response_synthesizer = get_response_synthesizer()

node_postprocessors = [KeywordNodePostprocessor(required_keywords=["Vehicles"], exclude_keywords=["cybertrucks"])]

query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer,
                                    node_postprocessors=node_postprocessors)


while True:
    question = input("Ask a question: ")
    response = query_engine.query(question)
    print(response)