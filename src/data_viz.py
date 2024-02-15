import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def data_viz(results_data):
    # Step 1: Convert results_data into a DataFrame
    data = {
        "chain_type": [],
        "eval_time": [],
        "tokens_used": []
    }

    for result in results_data:
        for eval in result.eval:
            data["chain_type"].append(result.chain_type)
            data["eval_time"].append(eval["time"])
            data["tokens_used"].append(eval["tokens_used"])

    df = pd.DataFrame(data)

    # Calculate averages
    avg_data = df.groupby('chain_type').mean().reset_index()

    # Step 2: Create a grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    # Setting the positions and width for the bars
    positions = np.arange(len(avg_data['chain_type']))
    bar_width = 0.35

    # Plotting both eval_time and tokens_used
    bars1 = ax.bar(positions - bar_width/2, avg_data['eval_time'], bar_width, label='Eval Time')
    bars2 = ax.bar(positions + bar_width/2, avg_data['tokens_used'], bar_width, label='Tokens Used')

    # Adding some final touches
    ax.set_xlabel('Chain Type')
    ax.set_ylabel('Average Values')
    ax.set_title('Average Evaluation Time and Tokens Used by Chain Type')
    ax.set_xticks(positions)
    ax.set_xticklabels(avg_data['chain_type'])
    ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("data_viz.png")
