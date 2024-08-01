from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

template = """
Question: {question}

Answer: Let's work this out in a step by step way sto be sure we have the right answer.

"""


prompt = PromptTemplate.from_template(template)

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path="llms/llama-2-7b-chat.Q2_K.gguf",
    temperature=0.75,
    max_tokens=2500,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,
)

question = """
Question: {A prose battle between William Shakespeare and William Wordsworth}
"""

print(llm.invoke(question))

