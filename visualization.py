# visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_histograms(df, numeric_cols):
    num_cols = len(numeric_cols)
    plt.figure(figsize=(15, num_cols * 3))
    for i, feature in enumerate(numeric_cols, 1):
        plt.subplot(num_cols, 3, i)
        sns.histplot(df[feature], kde=True)
        plt.title(f'Histogram of {feature}')
    plt.tight_layout()
    plt.show()

def visualize_outliers(df, numeric_cols, title_prefix):
    cols = 3
    rows = int(np.ceil(len(numeric_cols) / cols))
    plt.figure(figsize=(15, 5 * rows))
    for i, column in enumerate(numeric_cols):
        plt.subplot(rows, cols, i + 1)
        sns.boxplot(y=df[column])
        plt.title(f'{title_prefix} Outliers: {column}')
    plt.tight_layout()
    plt.show()
