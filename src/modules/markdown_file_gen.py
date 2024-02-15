def results_data_to_markdown_table(results_data_list):
    headers = ["Chain Type", "Eval Time", "Tokens Used", "Example Number", "Predicted Query", "Predicted Answer", "Answer", "Result"]
    markdown_table = "| " + " | ".join(headers) + " |\n"
    markdown_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"

    for data in results_data_list:
        for eval in data.eval:
            # Ensure every value is a string, handling None and ensuring dict values are properly formatted or avoided
            row = [
                data.chain_type,
                str(eval.get("time", "")),
                str(eval.get("tokens_used", "")),
                str(eval.get("example_number", "")),
                eval.get("query", ""),
                eval.get("predicted_answer", "") if eval.get("predicted_answer") is not None else "",
                eval.get("answer", ""),
                eval.get("result", "") if eval.get("result") is not None else ""
            ]
            markdown_table += "| " + " | ".join([str(item) for item in row]) + " |\n"
    
    return markdown_table


def write_markdown_table_to_file(markdown_table, filename):
    # Write the markdown table to the specified file
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(markdown_table)
    
    print(f"Markdown table successfully written to {filename}")