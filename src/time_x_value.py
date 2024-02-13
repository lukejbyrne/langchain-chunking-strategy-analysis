from dotenv import load_dotenv, find_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator
from langchain_openai.embeddings import OpenAIEmbeddings
import sys
import os
from datetime import datetime

def qa_analysis(llm, chain_type, retriever, verbose):
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type=chain_type, 
        retriever=retriever, 
        verbose=verbose
    )

    # record current timestamp
    start = datetime.now()

    response = qa.run(query)

    # record loop end timestamp
    end = datetime.now()

    # find difference loop start and end time and display
    td = (end - start).total_seconds() * 10**3
    print(f"Response: {response}\nThe time of execution of above program is : {td:.03f}ms")

    # return object with time and response?

# Basic Setup
_ = load_dotenv(find_dotenv()) # read local .env file
# add the parent directory of 'src' to sys.path for local module imports
# sys.path.insert(0, os.path.abspath('../modules/'))
from modules.set_model import llm_model
llm_model = llm_model()

# Load data into vector db
file = '../data/OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
documents = loader.load()
embedding = OpenAIEmbeddings() # defining embedding
db = Chroma.from_documents(
    documents, 
    embedding
)

# query =  "Please list all your shirts with sun protection in a table \
# in markdown and summarize each one."

query = "Please suggest a shirt with sunblocking" # test to reduce token use

#TODO: iterate and use dictionary?
#TODO: time how long query takes
#TODO: define criteria for measurement
#TODO: llm to create and evaluate

# layers vector db on llm to inform decisions and responses
llm = ChatOpenAI(temperature = 0.0, model=llm_model)
retriever = db.as_retriever()

qa_analysis(llm, "stuff", retriever, True)
qa_analysis(llm, "map_reduce", retriever, True)
qa_analysis(llm, "refine", retriever, True)
qa_analysis(llm, "map_rerank", retriever, True)
