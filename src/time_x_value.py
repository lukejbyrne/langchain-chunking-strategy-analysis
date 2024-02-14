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

def qa_analysis(llm, chain_type, retriever, verbose, query):
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

    start = datetime.now()

    # Measure number of tokens used
    with get_openai_callback() as cb:
        # Execute the QA analysis
        response = qa.run(query)
    print(cb.total_tokens)

    end = datetime.now()

    # Calculate the difference between the end and start timestamps to get the execution duration.
    # The duration is converted to milliseconds for a more precise and readable format.
    td = (end - start).total_seconds() * 10**3
    
    print(f"Response: {response}\nThe time of execution of above program is : {td:.03f}ms")

    # return object with time and response?

def main():
    # Basic Setup
    _ = load_dotenv(find_dotenv()) # read local .env file

    # Load data into vector db or use existing one
    file_path = '../data/OutdoorClothingCatalog_1000.csv'
    embedding = OpenAIEmbeddings()  # Define embedding

    # Check if vector DB exists for the CSV, and load or create accordingly
    db = check_and_load_vector_db(file_path, embedding)

    queries = ["Please suggest a shirt with sunblocking", "Please suggest a shirt with sunblocking and tell me why this one", "Please suggest three shirts with sunblocking and tell me why. Give this back to me in markdown code as a table", "Please suggest three shirts with sunblocking and tell me why. Give this back to me in markdown code as a table, with a summary below outlining why sunblocking is important"]
    #TODO: iterate and use dictionary?
    #TODO: define criteria for measurement

    # Configure LLM for querying
    # layers vector db on llm to inform decisions and responses
    llm = ChatOpenAI(temperature = 0.0, model=llm_model())
    retriever = db.as_retriever()

    # Run analysis
    # for i in queries:
    #     qa_analysis(llm, "stuff", retriever, True, i)
    #     qa_analysis(llm, "map_reduce", retriever, True, i)
    #     qa_analysis(llm, "refine", retriever, True, i)
    #     qa_analysis(llm, "map_rerank", retriever, True, i)

    #TODO: llm to create and evaluate
    # EVALUATION

    tuple = generate_qas(file_path, db, llm, "stuff")
    qa = tuple[0]
    examples = tuple[1]
    evaluate(qa, examples, llm)

if __name__ == '__main__':
    main()