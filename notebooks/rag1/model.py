import chromadb
import logging
import sys
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import (Settings, VectorStoreIndex, SimpleDirectoryReader, PromptTemplate)
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

global query_engine
query_engine = None

def init_llm():
    llm = Ollama(model="llama3", request_timeout=500.0)
    embed_model = OllamaEmbedding(    model_name="llama3",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},)

    Settings.llm = llm
    Settings.embed_model = embed_model


def init_index(embed_model):
    reader = SimpleDirectoryReader(input_dir="./docs2", recursive=True)
    documents = reader.load_data()

    logging.info("index creating with %d documents", len(documents))

    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("iollama")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)

    return index


def init_query_engine(index):
    global query_engine

    template = (
        "You are an AI expert.\n"
        "Here is some context related to the query:\n"
        "${context_str}\n"
        "Considering the above information, respond to the following query:\n"
        "Question: {query_str}\n\n"
        "Answer in detail, starting with the phrase:'According to our expertise,  '"
    )

    qa_template = PromptTemplate(template)

    query_engine = index.as_query_engine(text_qa_template = qa_template, similarity_top_k=3)

    return query_engine



def chat(input_question, user):
    global query_engine

    response = query_engine.query(input_question)
    logging.info("Got response from LLM - %s", response)

    return response.response


def chat_cmd():
    global query_engine

    while True:
        input_question = input("Enter your question ('exit' to quit):")
        if input_question.lower() == 'exit':
            break

        response = query_engine.query(input_question)
        logging.info("got response from llm - %s", response)


if __name__ == "__main__":
    init_llm()
    index = init_index(Settings.embed_model)
    init_query_engine(index)
    chat_cmd()
