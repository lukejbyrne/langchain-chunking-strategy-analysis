from dotenv import load_dotenv, find_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
import os
from datetime import datetime
from modules.set_model import llm_model
from langchain.callbacks import get_openai_callback
from modules.evaluation import generate_qas, evaluate
from modules.results_data import ResultsData

def vct_db_filename_gen(file_path):
    # Derive vector DB filename from CSV filename
    base_name = os.path.basename(file_path)
    db_file_name = os.path.splitext(base_name)[0] + ".vecdb"

    return os.path.join(os.path.dirname(file_path), db_file_name)

def check_and_load_vector_db(file_path, embedding):
    """
    Checks if a vector db file exists for the given file_path, 
    loads it if exists, otherwise creates it from the csv and saves it.
    """
    # Derive vector DB filename from CSV filename
    db_file_path = vct_db_filename_gen(file_path)

    # Check if the vector DB file exists
    if os.path.exists(db_file_path):
        print(f"Loading existing vector DB from {db_file_path}")
        db = Chroma(persist_directory=db_file_path, embedding_function=embedding)
    else:
        print(f"Vector DB not found. Creating from {file_path}")
        # Load the CSV and create the vector DB
        loader = CSVLoader(file_path=file_path)
        documents = loader.load()
        # Save the newly created vector DB
        db = Chroma.from_documents(documents, embedding, persist_directory=db_file_path)
        print(f"Saved new vector DB to {db_file_path}")
    
    return db

def qa_analysis(llm, chain_type, retriever, verbose, query, results_data):
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
        
        # Execute the QA analysis
        response = qa.run(query)
        
        end = datetime.now()
    
    tokens_used = cb.total_tokens

    # Calculate the difference between the end and start timestamps to get the execution duration.
    # The duration is converted to milliseconds for a more precise and readable format.
    td = (end - start).total_seconds() * 10**3
    
    print(f"Response: {response}\nThe time of execution of above program is : {td:.03f}ms")

    for item in results_data:
        if item['chain_type'] == chain_type:
            # Update the existing dictionary
            item.update({'time': td, 'tokens': tokens_used})
            break
        else:
            # Append a new instance of ResultsData if no matching chain_type was found
            results_data.append({'chain_type': chain_type, 'time': td, 'tokens': tokens_used})

    print("\n\nTESTING\n:" + '\n'.join([str(item) for item in results_data]))

    return results_data


def results_data_to_markdown_table(results_data_list):
    # Define the header of the markdown table
    headers = ["Type", "Time", "Tokens", "Eval i", "Pred Query", "Pred Answer", "Answer", "Result"]
    # Create the markdown table header and separator rows
    markdown_table = "| " + " | ".join(headers) + " |\n"
    markdown_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    
    # Iterate over each ResultsData instance and add its data to the table
    for data in results_data_list:
        row = [
            data.type,
            str(data.time),
            str(data.tokens) if data.tokens is not None else "",
            str(data.eval["i"]) if data.eval["i"] is not None else "",
            data.eval["pred_query"] if data.eval["pred_query"] is not None else "",
            data.eval["pred_answer"] if data.eval["pred_answer"] is not None else "",
            data.eval["answer"] if data.eval["answer"] is not None else "",
            data.eval["result"] if data.eval["result"] is not None else "",
        ]
        markdown_table += "| " + " | ".join(row) + " |\n"
    
    return markdown_table

def write_markdown_table_to_file(markdown_table, file_name):
    """
    Writes the given markdown table string to a file.

    Parameters:
    - markdown_table: A string containing the markdown table to write to the file.
    - file_name: The name of the file to write the markdown table to.
    """
    with open(file_name, 'w') as file:
        file.write(markdown_table)

    # Specify the file name
    file_name = 'results_table.md'

    # Write the markdown table to the file
    write_markdown_table_to_file(markdown_table, file_name)


def main():
    # Basic Setup
    _ = load_dotenv(find_dotenv()) # read local .env file
    results_data = []

    # Load data into vector db or use existing one
    file_path = '../data/OutdoorClothingCatalog_1000.csv'
    embedding = OpenAIEmbeddings()  # Define embedding

    # Check if vector DB exists for the CSV, and load or create accordingly
    db = check_and_load_vector_db(file_path, embedding)

    queries = ["Please suggest a shirt with sunblocking", "Please suggest a shirt with sunblocking and tell me why this one", "Please suggest three shirts with sunblocking and tell me why. Give this back to me in markdown code as a table", "Please suggest three shirts with sunblocking and tell me why. Give this back to me in markdown code as a table, with a summary below outlining why sunblocking is important"]

    # Configure LLM for querying
    # layers vector db on llm to inform decisions and responses
    llm = ChatOpenAI(temperature = 0.0, model=llm_model())
    retriever = db.as_retriever()

    # Run analysis
    for i in queries:
        results_data = qa_analysis(llm, "stuff", retriever, True, i, results_data)
        results_data = qa_analysis(llm, "map_reduce", retriever, True, i, results_data)
        results_data = qa_analysis(llm, "refine", retriever, True, i, results_data)
        results_data = qa_analysis(llm, "map_rerank", retriever, True, i, results_data)

    # Generate evaluation Q&As
    tuple = generate_qas(file_path, db, llm, "stuff") #TODO: change this?
    qa = tuple[0]
    examples = tuple[1]

    # Evaluate 
    results_data = evaluate(qa, examples, llm. results_data)

    # Generate results in markdown
    md_table = results_data_to_markdown_table(results_data)

    # Write results to file
    write_markdown_table_to_file(md_table, "results.md")

if __name__ == '__main__':
    main()