import os
from dotenv import load_dotenv, find_dotenv
from tutorials.set_model import llm_model
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain.llms import OpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings

_ = load_dotenv(find_dotenv()) # read local .env file
llm_model = llm_model()

file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

query ="Please list all your shirts with sun protection \
in a table in markdown and summarize each one."

# Step By Step

loader = CSVLoader(file_path=file)

docs = loader.load()

docs[0]

embeddings = OpenAIEmbeddings()

embed = embeddings.embed_query("Hi my name is Harrison")

print(len(embed))

print(embed[:5])

db = DocArrayInMemorySearch.from_documents(
    docs, 
    embeddings
)

query = "Please suggest a shirt with sunblocking"

docs = db.similarity_search(query)

len(docs)

docs[0]

# hand cranked version
llm = ChatOpenAI(temperature = 0.0, model=llm_model)
qdocs = "".join([docs[i].page_content for i in range(len(docs))])
response = llm.call_as_llm(f"{qdocs} Question: Please list all your \
shirts with sun protection in a table in markdown and summarize each one.")
display(Markdown(response))

# langchain version - saves pulling the docs together into a single var as uses vector db
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
display(Markdown(response))

# shortened langchain version (equivalent to above)
response = index.query(query, llm=llm)

# customise the index and vector store
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings,
).from_loaders([loader])

# above is the stuff method
#   add into one big doc
#   pro: single simple call
#   con: large docs / many docs may have a prompt that exceeds LLM context
#
# to solve
#   1. map reduce
#       sends each chunk to an LLM (parallel, treats each doc as seperate)
#       then sends these responses to LLM 
#       common use: summarise
#   2. refine
#       iteratively loops over all docs
#       building on each answer each time
#       will take longer and have larger answer
#   3. map rerank
#       map reduce but gives back a score
#       highest score wins
#       relies on LLM knowing how to score