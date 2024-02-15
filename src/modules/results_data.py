class ResultsData:
    def __init__(self, chain_type, time=None, tokens_used=None, example_number=None, predicted_query=None, predicted_answer=None, answer=None, result=None):
        self.chain_type = chain_type
        self.eval = []
        if example_number is not None:
            self.append_evaluation(time, tokens_used, example_number, predicted_query, answer, predicted_answer, result)
    
    def append_evaluation(self, time, tokens_used, example_number, predicted_query, answer, predicted_answer, result):
        """Append a new evaluation result to the eval list."""
        self.eval.append({
            "time": time,
            "tokens_used": tokens_used,
            "example_number": example_number,
            "query": predicted_query,
            "predicted_answer": predicted_answer,
            "answer": answer,
            "result": result
        })
