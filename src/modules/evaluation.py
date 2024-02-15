from langchain.evaluation.qa import QAGenerateChain
from langchain.evaluation.qa import QAEvalChain
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import CSVLoader
from modules.set_model import llm_model
from langchain_openai import ChatOpenAI
from modules.results_data import ResultsData

def langchain_output_parser(qa_output):
    """
    Transforms the QA output from langchain into a dictionary format without the 'qa_pairs' field.
    
    Parameters:
    - qa_output: A list of dictionaries, where each dictionary contains 'qa_pairs' among other possible fields.

    Returns:
    - A list of dictionaries, where each dictionary directly contains 'query' and 'answer' fields.
    """
    parsed_output = []
    for item in qa_output:
        # Assuming each item in qa_output is a dictionary with a 'qa_pairs' key
        qa_pair = item.get('qa_pairs', {})
        # Repackage the qa_pair without the 'qa_pairs' field
        reformatted_item = {
            'query': qa_pair.get('query', ''),
            'answer': qa_pair.get('answer', '')
        }
        parsed_output.append(reformatted_item)
    return parsed_output


def generate_qas(file_path, db, llm, chain_type):
    # Load vector db to index
    loader = CSVLoader(file_path=file_path)
    data = loader.load()
    index = VectorStoreIndexWrapper(vectorstore=db)

    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type=chain_type, 
        retriever=index.vectorstore.as_retriever(), 
        verbose=True,
        # chain_type_kwargs = {
        #     "document_separator": "<<<<>>>>>"
        # }
    ) 

    # LLM-Generated example Q&A pairs 
    example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI(model=llm_model()))
    # the warning below can be safely ignored
    raw_examples = example_gen_chain.apply( # create raw examples
        [{"doc": t} for t in data[:5]],
    )

    # Parse the raw examples into required format
    examples = langchain_output_parser(raw_examples)

    # run for manual evaluation
    qa.run(examples[0]["query"])

    return qa, examples

def evaluate(chain_type, qa, examples, llm, results_data):
    # LLM assisted evaluation
    predictions = qa.apply(examples)
    eval_chain = QAEvalChain.from_llm(llm)
    graded_outputs = eval_chain.evaluate(examples, predictions)

    # turn to object and return
    # using llm as real answer and predicted answer are not similar in a string match sense, e.g. look at example_llm_eval.txt
    for i, eg in enumerate(examples):
        
        example_number = i
        query = predictions[i]['query']
        answer = predictions[i]['answer']
        predicted_answer = predictions[i]['result']
        result = graded_outputs[i]['results']
        
        print(f"Example {example_number}:")
        print("Question: " + query)
        print("Real Answer: " + answer)
        print("Predicted Answer: " + predicted_answer)
        print("Predicted Grade: " + result)
        print()

        results_data = add_to_results_list(results_data, chain_type, query, example_number=i, predicted_answer=predicted_answer, answer=answer, result=result)
    return results_data

def add_to_results_list(results_data, chain_type, query, td=None, tokens_used=None, number=None, response=None, answer=None, result=None):
    found = False
    for item in results_data:
        if item.chain_type == chain_type:
            # Update the existing dictionary
            item.append_evaluation(time=td, tokens_used=tokens_used, example_number=number, 
                         predicted_query=query, answer=response, predicted_answer=answer, result=result)
            found = True
            break

    if not found:
        # Append a new instance of ResultsData if no matching chain_type was found
        results_data.append(ResultsData(chain_type=chain_type, time=td, tokens_used=tokens_used, 
                                        example_number=number, predicted_query=query, 
                                        answer=response, predicted_answer=answer, 
                                        result=result))
        
    return results_data