import langchain
from langchain.evaluation.qa import QAGenerateChain
from langchain.evaluation.qa import QAEvalChain
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import CSVLoader
from modules.set_model import llm_model
from langchain_openai import ChatOpenAI

def generate_qas(file_path, db, llm):
    # Load vector db to index
    loader = CSVLoader(file_path=file_path)
    data = loader.load()
    index = VectorStoreIndexWrapper(vectorstore=db)

    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=index.vectorstore.as_retriever(), 
        verbose=True,
        chain_type_kwargs = {
            "document_separator": "<<<<>>>>>"
        }
    ) 

    #TODO: LLM generated Q&As?
    # LLM-Generated example Q&A pairs 
    example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI(model=llm_model()))
    # the warning below can be safely ignored
    examples = example_gen_chain.apply_and_parse(
        [{"doc": t} for t in data[:5]]
    )
    print(examples[0])
    print(data[0])

    # run example
    qa.run(examples[0]["query"]) #TODO: PRIORITY; key error?

    return qa, examples

def evaluate(qa, examples, llm):
    # LLM assisted evaluation
    # How are we going to evaulate those created by LLM?
    predictions = qa.apply(examples)
    eval_chain = QAEvalChain.from_llm(llm)
    graded_outputs = eval_chain.evaluate(examples, predictions)

    # turn to object and return
    # using llm as real answer and predicted answer are not similar in a string match sense, e.g. look at example_llm_eval.txt
    for i, eg in enumerate(examples):
        print(f"Example {i}:")
        print("Question: " + predictions[i]['query'])
        print("Real Answer: " + predictions[i]['answer'])
        print("Predicted Answer: " + predictions[i]['result'])
        print("Predicted Grade: " + graded_outputs[i]['text'])
        print()

    print(graded_outputs[0])

    # store in file for manual evaluation
    # Manual Evaluation
    langchain.debug = True
    qa.run(examples[0]["query"]) # example output at example_output.txt
    # Turn off the debug mode
    langchain.debug = False