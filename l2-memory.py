import os
from dotenv import load_dotenv, find_dotenv
import warnings
from set_model import llm_model
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationTokenBufferMemory
from langchain.llms import OpenAI
llm = ChatOpenAI(temperature=0.0, model=llm_model)

_ = load_dotenv(find_dotenv()) # read local .env file
warnings.filterwarnings('ignore')
llm_model = llm_model()

llm = ChatOpenAI(temperature=0.0, model=llm_model)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True # To see what LangChain is actually doing (see the prompt)
)

conversation.predict(input="Hi, my name is Andrew")
conversation.predict(input="What is 1+1?")
conversation.predict(input="What is my name?")
print(memory.buffer) # See convo so far
memory.load_memory_variables({}) # prints all mem vars, in this instance it only shows 'history' (as this is the previosu convo)

# To explicitly clear and create a memory, therefore stateless as we just give context each time to give a state feel
memory = ConversationBufferMemory()
memory.save_context({"input": "Hi"}, 
                    {"output": "What's up"})
print(memory.buffer)
memory.save_context({"input": "Not much, just hanging"}, 
                    {"output": "Cool"})
print(memory.buffer)

memory.load_memory_variables({})

# ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(k=1) # where k = # of msgs to remmeber each (1 ai and 1 user) 

memory.save_context({"input": "Hi"},
                    {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})

memory.load_memory_variables({})

llm = ChatOpenAI(temperature=0.0, model=llm_model)
memory = ConversationBufferWindowMemory(k=1)
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=False
)

memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=50)
memory.save_context({"input": "AI is what?!"},
                    {"output": "Amazing!"})
memory.save_context({"input": "Backpropagation is what?"},
                    {"output": "Beautiful!"})
memory.save_context({"input": "Chatbots are what?"}, 
                    {"output": "Charming!"})

memory.load_memory_variables({})