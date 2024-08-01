from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

template = """
Question: {question}

Answer: Give the accurate answer. If you are unsure say you do not know.
"""
prompt = PromptTemplate.from_template(template)

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path="llms/llama-2-7b-chat.Q2_K.gguf",
    temperature=0.8,
    max_tokens=2500,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True
)

llm_chain = prompt | llm

question = """
Question: Write a Python code that takes an input and calculates the next 100 fibonacci series
"""

print(llm_chain.invoke(question))