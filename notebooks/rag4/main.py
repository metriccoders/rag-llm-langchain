from llama_index.llms.openai import OpenAI
import os

os.environ["OPENAI_API_KEY"] = ""


response = OpenAI().complete("Metriccoders website is ")
print(response)