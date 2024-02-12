import warnings
import os
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from set_model import llm_model
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import pandas as pd
from langchain.chains import SimpleSequentialChain

# Setup
warnings.filterwarnings('ignore')
_ = load_dotenv(find_dotenv()) # read local .env file
llm = ChatOpenAI(temperature=0.0, model=llm_model)

df = pd.read_csv('Data.csv')

#LLMChain
llm = ChatOpenAI(temperature=0.9, model=llm_model)
prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe \
    a company that makes {product}?"
)
chain = LLMChain(llm=llm, prompt=prompt)
product = "Queen Size Sheet Set"
chain.run(product)

#SimpleSequentialChain - 1 in 1 out
llm = ChatOpenAI(temperature=0.9, model=llm_model)

# prompt template 1
first_prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe \
    a company that makes {product}?"
)

# Chain 1
chain_one = LLMChain(llm=llm, prompt=first_prompt)

# prompt template 2
second_prompt = ChatPromptTemplate.from_template(
    "Write a 20 words description for the following \
    company:{company_name}"
)
# chain 2
chain_two = LLMChain(llm=llm, prompt=second_prompt)

overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two],
                                             verbose=True
                                            )