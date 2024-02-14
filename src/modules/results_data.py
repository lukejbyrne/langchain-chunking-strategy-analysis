class ResultsData:
    def __init__(self, chain_type, time, tokens_used=None, example_number=None, predicted_query=None, predicted_answer=None, answer=None, result=None):
        self.chain_type = chain_type
        self.time = time
        self.tokens_used = tokens_used
        self.eval = {
            "example_number": example_number,
            "query": predicted_query,
            "predicted_answer": predicted_answer,
            "answer": answer,
            "result": result
        }
        