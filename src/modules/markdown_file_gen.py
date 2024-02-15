def results_data_to_markdown_table(results_data_list):
    # Define the header of the markdown table
    headers = ["Chain Type", "Eval Time", "Tokens Used", "Example Number", "Predicted Query", "Predicted Answer", "Answer", "Result"]
    # Create the markdown table header and separator rows
    markdown_table = "| " + " | ".join(headers) + " |\n"
    markdown_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    
    # Iterate over each ResultsData instance
    for data in results_data_list:
        # And then iterate over each evaluation within the ResultsData instance
        for eval in data.eval:
            # Construct each row with the appropriate data
            row = [
                data.chain_type,  # Corrected from data.type to data.chain_type
                str(eval.get("time", "")),  # Using .get() for safer access to dictionary keys
                str(eval.get("tokens_used", "")),
                str(eval.get("example_number", "")),
                eval.get("query", ""),
                eval.get("predicted_answer", ""),
                eval.get("answer", ""),
                eval.get("result", "")
            ]
            markdown_table += "| " + " | ".join(row) + " |\n"
    
    return markdown_table

def write_markdown_table_to_file(markdown_table, filename):
    # Write the markdown table to the specified file
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(markdown_table)
    
    print(f"Markdown table successfully written to {filename}")