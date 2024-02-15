import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors

# Assuming results_data is your list of ResultsData objects

def data_viz(results_data):
    # Convert results_data into a DataFrame
    data = {
        "chain_type": [],
        "eval_time": [],
        "tokens_used": [],
        "is_correct": []
    }

    for result in results_data:
        for eval in result.eval:
            data["chain_type"].append(result.chain_type)
            data["eval_time"].append(eval["time"])
            data["tokens_used"].append(eval["tokens_used"])
            # 1 for correct, 0 otherwise
            data["is_correct"].append(1 if eval["result"] == "CORRECT" else 0)

    df = pd.DataFrame(data)

    # Calculate averages and correctness ratio
    agg_data = df.groupby('chain_type').agg({
        'eval_time': 'mean',
        'tokens_used': 'mean',
        'is_correct': lambda x: np.mean(x) * 100  # correctness percentage
    }).reset_index()

    # Create a grouped bar chart with an additional bar for correct answers
    fig, ax = plt.subplots(figsize=(12, 8))

    # Normalize correctness percentage for color mapping
    norm = mcolors.Normalize(vmin=agg_data['is_correct'].min(), vmax=agg_data['is_correct'].max())
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm)
    sm.set_array([])

    positions = np.arange(len(agg_data))
    bar_width = 0.35

    # Evaluation Time bars with color based on correctness ratio
    for idx, row in agg_data.iterrows():
        ax.bar(idx - bar_width/2, row['eval_time'], bar_width, color=sm.to_rgba(row['is_correct']))

    # Tokens Used bars
    ax.bar(positions + bar_width/2, agg_data['tokens_used'], bar_width, color='lightblue')

    # Correctness adjustment: create a dummy imshow for the colorbar reference
    cb_ax = fig.add_axes([0, 0, 0.1, 0.1], visible=False)
    cb_im = cb_ax.imshow([[0, 100]], cmap="RdYlGn")
    cb_im.set_clim(vmin=0, vmax=100)
    cbar = fig.colorbar(cb_im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label('Correctness Ratio (%)')

    # Final touches
    ax.set_xlabel('Chain Type')
    ax.set_ylabel('Average Values')
    ax.set_title('Evaluation Time and Tokens Used by Chain Type with Correctness Ratio')
    ax.set_xticks(positions)
    ax.set_xticklabels(agg_data['chain_type'])
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig("../results/data_viz.png")
