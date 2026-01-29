import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_metrics(df, title):
    df_melted = df.melt(id_vars="Variant", value_vars=["Hit Rate", "MRR"], var_name="Metric", value_name="Score")

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=df_melted, x="Variant", y="Score", hue="Metric", palette="viridis")

    plt.title(title, fontsize=14)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)

    plt.show()

def plot_extended_metrics(df):
    metrics_list = [
        'faithfulness',
        'citation_correctness',
        'answer_relevance',
        'semantic_similarity'
    ]

    scores = {m: df[m].mean() for m in metrics_list}

    df_plot = pd.DataFrame(list(scores.items()), columns=['Metric', 'Score'])

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_plot, x='Metric', y='Score', palette='rocket')

    plt.ylim(0, 1.1)
    plt.title("Comprehensive Generation Quality", fontsize=14)
    plt.grid(axis='y', alpha=0.3)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)

    plt.show()