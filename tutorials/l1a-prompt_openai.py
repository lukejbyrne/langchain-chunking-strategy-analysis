import os
from set_model import llm_model
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

_ = load_dotenv(find_dotenv()) # read local .env file

llm_model = llm_model()

def get_completion(prompt, model=llm_model):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(model=model,
    messages=messages,
    temperature=0)
    return response.choices[0].message.content

get_completion("What is 1+1?")

customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

style = """American English \
in a calm and respectful tone
"""

prompt = f"""Translate the text \
that is delimited by triple backticks 
into a style that is {style}.
text: ```{customer_email}```
"""

print(prompt)

response = get_completion(prompt)

response
