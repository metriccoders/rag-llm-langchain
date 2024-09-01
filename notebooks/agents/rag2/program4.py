import os
os.environ["OPENAI_API_KEY"] = ""

from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.evaluation import FaithfulnessEvaluator


llm = OpenAI(model="gpt-4", temperature=0.1)

data_path = "./data"

documents = SimpleDirectoryReader(data_path).load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

evaluator = FaithfulnessEvaluator(llm=llm)

response = query_engine.query("What was the total revenue in 2022?")
print(response)

eval_result = evaluator.evaluate_response(response=response)
print(str(eval_result.passing))

from llama_index.core.evaluation import RetrieverEvaluator
retriever = index.as_retriever(similarity_top_k=2)
retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"], retriever=retriever
)
retriever_evaluator.evaluate(
    query="query", expected_ids=["node_id1", "node_id2"]
)

