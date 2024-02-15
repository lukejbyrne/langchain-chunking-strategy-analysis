from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from modules.set_model import llm_model
from modules.evaluation import generate_qas, evaluate
from modules.markdown_file_gen import results_data_to_markdown_table, write_markdown_table_to_file
from modules.vector_db import check_and_load_vector_db
from modules.qa_analysis import qa_analysis

def main():
    # Basic Setup
    _ = load_dotenv(find_dotenv()) # read local .env file
    results_data = []
    strategies = ["stuff", "map_reduce", "refine", "map_rerank"]

    # Load data into vector db or use existing one
    file_path = 'data/OutdoorClothingCatalog_1000.csv'
    embedding = OpenAIEmbeddings()  # Define embedding

    # Check if vector DB exists for the CSV, and load or create accordingly
    db = check_and_load_vector_db(file_path, embedding)

    queries = ["Please suggest a shirt with sunblocking", "Please suggest a shirt with sunblocking and tell me why this one", "Please suggest three shirts with sunblocking and tell me why. Give this back to me in markdown code as a table", "Please suggest three shirts with sunblocking and tell me why. Give this back to me in markdown code as a table, with a summary below outlining why sunblocking is important"]

    # Configure LLM for querying
    # layers vector db on llm to inform decisions and responses
    llm = ChatOpenAI(temperature = 0.0, model=llm_model())
    retriever = db.as_retriever()

    # Manual analysis - TODO: add answers
    # for index, query in enumerate(queries, start=1):
    #     results_data = qa_analysis(llm, "stuff", retriever, True, query, index, results_data)
    #     results_data = qa_analysis(llm, "map_reduce", retriever, True, query, index, results_data)
    #     results_data = qa_analysis(llm, "refine", retriever, True, query, index, results_data)
    #     results_data = qa_analysis(llm, "map_rerank", retriever, True, query, index, results_data)

    # LLM QA Gen AND Evaluate
    for strat in strategies:
        # Generate evaluation Q&As
        tuple = generate_qas(file_path, db, llm, strat) #TODO: change this?
        qa = tuple[0]
        examples = tuple[1]

        # Evaluate 
        results_data = evaluate(strat, qa, examples, llm, results_data)

    # Generate results in markdown
    md_table = results_data_to_markdown_table(results_data)

    # Write results to file
    write_markdown_table_to_file(md_table, "results.md")

if __name__ == '__main__':
    main()