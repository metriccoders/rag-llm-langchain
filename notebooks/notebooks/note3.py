from langchain_community.llms.llamafile import Llamafile

llm = Llamafile()
print(llm.invoke("Just tell a joke about ChatGPT"))
