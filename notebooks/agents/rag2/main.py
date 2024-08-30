import os
os.environ["OPENAI_API_KEY"] = ""

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool


Settings.llm = OpenAI(model="gpt-4", temperature=0)

data_path = "../../../datasets/data"

documents = SimpleDirectoryReader(data_path).load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

from llama_index.core.tools import QueryEngineTool
def multiply(a: float, b:float) -> float:
    """
    Multiple two numbers and returns the product
    :param a:
    :param b:
    :return:
    """

    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)

tesla_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="tesla_earnings_2022",
    description="A RAG engine with some basic facts about Tesla 2022 Earnings"
)

agent = ReActAgent.from_tools([
    multiply_tool,
    tesla_tool
], verbose=True)

response = agent.chat("What is the revenue in 2022 multiplied by 4?")
print(response)