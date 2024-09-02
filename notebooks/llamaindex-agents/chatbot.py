import os

os.environ["OPENAI_API_KEY"] = ""

import nest_asyncio
import nltk
nltk.download('averaged_perceptron_tagger')
nest_asyncio.apply()

from llama_index.readers.file import UnstructuredReader
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from pathlib import Path

years = [2022, 2021, 2020, 2019]
doc_set = {}
all_docs = []

loader = UnstructuredReader()
for year in years:
    year_docs = loader.load_data(file=Path(f"./data/uber/UBER_{year}.html"), split_documents=False)
    for d in year_docs:
        d.metadata = {"year": year}
    doc_set[year] = year_docs
    all_docs.extend(year_docs)

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

Settings.chunk_size = 512
Settings.chunk_overlap = 64
Settings.llm = OpenAI(model="gpt-4")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

index_set = {}
for year in years:
    storage_context = StorageContext.from_defaults()
    cur_index = VectorStoreIndex.from_documents(
        doc_set[year],
        storage_context=storage_context
    )

    index_set[year] = cur_index
    storage_context.persist(persist_dir=f"./storage/{year}")

from llama_index.core import load_index_from_storage

index_set = {}
for year in years:
    storage_context = StorageContext.from_defaults(
        persist_dir=f"./storage/{year}"
    )
    cur_index = load_index_from_storage(storage_context=storage_context)

    index_set[year] = cur_index

individual_query_engine_tools = [
    QueryEngineTool(
        query_engine=index_set[year].as_query_engine(),
        metadata=ToolMetadata(
            name=f"vector_index_{year}",
            description=("useful for when you want to answer queries about the"
                         f" {year} SEC 10-K for Uber"
                         ),
        ),
    )
    for year in years
]

from llama_index.core.query_engine import SubQuestionQueryEngine

query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=individual_query_engine_tools
)

query_engine_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="sub_question_query_engine",
        description=(
            "useful for when you want to answer queries that require analyzing"
            " multiple SEC 10-K documents for Uber"
        )
    )
)

tools = individual_query_engine_tools + [query_engine_tool]

from llama_index.agent.openai import OpenAIAgent
agent = OpenAIAgent.from_tools(tools, verbose=True)

response = agent.chat("Hey! I'm Suhas")
print(response)
