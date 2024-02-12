import warnings
import os
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from set_model import llm_model
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import pandas as pd

# Setup
warnings.filterwarnings('ignore')
_ = load_dotenv(find_dotenv()) # read local .env file
llm = ChatOpenAI(temperature=0.0, model=llm_model)

df = pd.read_csv('Data.csv')

llm = ChatOpenAI(temperature=0.9, model=llm_model)
prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe \
    a company that makes {product}?"
)
chain = LLMChain(llm=llm, prompt=prompt)
product = "Queen Size Sheet Set"
chain.run(product)
