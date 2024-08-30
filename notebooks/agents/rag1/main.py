from dotenv import load_dotenv
load_dotenv()

from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool

import os
os.environ["OPENAI_API_KEY"] = ""
llm = OpenAI(temperature=0.2, model="gpt-4")

def multiply(a: float, b:float) -> float:
    """
    Multiple two numbers and returns the product
    :param a:
    :param b:
    :return:
    """

    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)


def add(a: float, b: float) -> float:
    """
    Add two numbers and returns the sum
    :param a:
    :param b:
    :return:
    """
    return a+b


add_tool = FunctionTool.from_defaults(fn=add)

agent = ReActAgent.from_tools([multiply_tool, add_tool], llm=llm, verbose=True)
response = agent.chat("What is 100 + (100*4) ? Use a tool to calculate every step.")

print(response)