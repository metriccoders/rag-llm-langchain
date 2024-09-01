from dotenv import load_dotenv
load_dotenv()
import os
os.environ["OPENAI_API_KEY"] = ""
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.tools import QueryEngineTool

data_path = "./data"
Settings.llm = OpenAI(model="gpt-4", temperature=0)

def multiply(a: float, b:float) -> float:
    """
    Multiple two numbers and returns the product
    :param a:
    :param b:
    :return:
    """

    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)
def add(a: float, b:float) -> float:
    """
    Add two numbers and returns the product
    :param a:
    :param b:
    :return:
    """

    return a + b

add_tool = FunctionTool.from_defaults(fn=add)

documents = SimpleDirectoryReader(data_path).load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

tesla_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="tesla_earnings_tool_2022",
    description="A RAG Engine"
)

agent = ReActAgent.from_tools([multiply_tool, add_tool, tesla_tool], verbose=True)

response = agent.chat("What was the sales revenue in 2022?")
print(response)

response = agent.chat("What was the sales revenue in 2021?")
print(response)

response = agent.chat("What was the sum of the previous two responses?. Use a tool to answer")
print(response)