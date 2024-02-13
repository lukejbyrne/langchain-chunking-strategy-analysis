from dotenv import load_dotenv, find_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator
from langchain_openai.embeddings import OpenAIEmbeddings
import sys
import os

# Basic Setup
_ = load_dotenv(find_dotenv()) # read local .env file
# add the parent directory of 'src' to sys.path for local module imports
sys.path.insert(0, os.path.abspath('../modules/'))
from set_model import llm_model
llm_model = llm_model()

# Load data into vector db
file = '../../data/OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
documents = loader.load()
embedding = OpenAIEmbeddings() # defining embedding
db = Chroma.from_documents(
    documents, 
    embedding
)

# Run query against the db
query = "Please suggest a shirt with sunblocking"
docs = db.similarity_search(query)
print(len(docs))
print(docs[0])

# langchain version - saves pulling the docs together into a single var as uses vector db
llm = ChatOpenAI(temperature = 0.0, model=llm_model)
retriever = db.as_retriever()
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)
query =  "Please list all your shirts with sun protection in a table \
in markdown and summarize each one."
response = qa_stuff.run(query)
print("here")
print(response)

# customise the index and vector store - why?
index = VectorstoreIndexCreator(
    vectorstore_cls=Chroma,
    embedding=embedding,
).from_loaders([loader])
