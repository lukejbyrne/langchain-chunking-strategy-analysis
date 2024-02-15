from datetime import datetime
from langchain.chains import RetrievalQA
from langchain.callbacks import get_openai_callback
from modules.results_data import ResultsData
from evaluation import add_to_results_list

def qa_analysis(llm, chain_type, retriever, verbose, query, number, results_data):
    """
    Initializes a QA analysis with a given language model, chain type, and retriever.
    Then, it runs the QA analysis, timing its execution and printing the response along with the execution time.
    """
    # Initialize the RetrievalQA object with the specified parameters.
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type=chain_type, 
        retriever=retriever, 
        verbose=verbose
    )

    # Measure number of tokens used
    with get_openai_callback() as cb:
        start = datetime.now()

        try:
            # Execute the QA analysis
            response = qa.invoke(query) #TODO: i've only added queries, no answers...
        except ValueError as e: 
            response = e

        end = datetime.now()
    
    tokens_used = cb.total_tokens

    # Calculate the difference between the end and start timestamps to get the execution duration.
    # The duration is converted to milliseconds for a more precise and readable format.
    td = (end - start).total_seconds() * 10**3
    
    print(f"Response: {response}\nThe time of execution of above program is : {td:.03f}ms")

    results_data = add_to_results_list(results_data, chain_type, query, td, tokens_used, number, response)

    print("\n\nTESTING\n:" + '\n'.join([str(item) for item in results_data]))

    return results_data
