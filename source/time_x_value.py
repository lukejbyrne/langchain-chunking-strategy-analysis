import os
from tutorials.set_model import llm_model
from dotenv import load_dotenv, find_dotenv
import langchain
from langchain.evaluation.qa import QAGenerateChain
from langchain.evaluation.qa import QAEvalChain

_ = load_dotenv(find_dotenv()) # read local .env file
llm_model = llm_model()

# Create our QandA application

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores import DocArrayInMemorySearch

file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
data = loader.load()

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

llm = ChatOpenAI(temperature = 0.0, model=llm_model)
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=index.vectorstore.as_retriever(), 
    verbose=True,
    chain_type_kwargs = {
        "document_separator": "<<<<>>>>>"
    }
)

# Coming up with test datapoints

print(data[10])
print(data[11])

# Hard-coded query examples

# examples = [
#     {
#         "query": "Do the Cozy Comfort Pullover Set\
#         have side pockets?",
#         "answer": "Yes"
#     },
#     {
#         "query": "What collection is the Ultra-Lofty \
#         850 Stretch Down Hooded Jacket from?",
#         "answer": "The DownTek collection"
#     }
# ]

# # LLM-Generated example Q&A pairs 

# example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI(model=llm_model))
# # the warning below can be safely ignored
# new_examples = example_gen_chain.apply_and_parse(
#     [{"doc": t} for t in data[:5]]
# )
# new_examples[0]
# data[0]

# # Combine examples
# examples += new_examples
# qa.run(examples[0]["query"])

# # Manual Evaluation
# langchain.debug = True
# qa.run(examples[0]["query"]) # example output at example_output.txt
# # Turn off the debug mode
# langchain.debug = False

# # LLM assisted evaluation
# # How are we going to evaulate those created by LLM?
# predictions = qa.apply(examples)
# llm = ChatOpenAI(temperature=0, model=llm_model)
# eval_chain = QAEvalChain.from_llm(llm)
# graded_outputs = eval_chain.evaluate(examples, predictions)

# # using llm as real answer and predicted answer are not similar in a string match sense, e.g. look at example_llm_eval.txt
# for i, eg in enumerate(examples):
#     print(f"Example {i}:")
#     print("Question: " + predictions[i]['query'])
#     print("Real Answer: " + predictions[i]['answer'])
#     print("Predicted Answer: " + predictions[i]['result'])
#     print("Predicted Grade: " + graded_outputs[i]['text'])
#     print()

# graded_outputs[0]