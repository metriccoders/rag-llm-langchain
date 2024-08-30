from dotenv import load_dotenv
load_dotenv()
import os
os.environ["OPENAI_API_KEY"] = ""
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_parse import LlamaParse

Settings.llm = OpenAI(model="gpt-4", temperature=0)

data_path = "./data"

def multiply(a: float, b:float) -> float:
    """
    Multiple two numbers and returns the product
    :param a:
    :param b:
    :return:
    """

    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)

documents = SimpleDirectoryReader(data_path).load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query("What is the revenue in 2022 multiplied by 4?")
print(response)

documents2 = LlamaParse( result_type="markdown").load_data(
    "./data/tesla-2022-earnings.pdf")
index2 = VectorStoreIndex.from_documents(documents2)
query_engine2 = index2.as_query_engine()

response2 = query_engine2.query("What is the revenue in 2022 multiplied by 4?")
print(response2)
