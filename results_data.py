class ResultsData:
    def __init__(self, type, time, tokens=None, eval_i=None, pred_query=None, pred_answer=None, answer=None, result=None):
        self.type = type
        self.time = time
        self.tokens = tokens
        self.eval = {
            "i": eval_i,
            "pred_query": pred_query,
            "pred_answer": pred_answer,
            "answer": answer,
            "result": result
        }
        