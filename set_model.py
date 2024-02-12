import datetime

# account for deprecation of LLM model

def llm_model():
    # Get the current date
    current_date = datetime.datetime.now().date()

    # Define the date after which the model should be set to "gpt-3.5-turbo"
    target_date = datetime.date(2024, 6, 12)

    # Set the model variable based on the current date
    if current_date > target_date:
        return "gpt-3.5-turbo"
    else:
        return "gpt-3.5-turbo-0301"
