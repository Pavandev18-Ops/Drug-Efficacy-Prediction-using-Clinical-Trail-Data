# visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_histograms(df, numeric_cols):
    """
    Plots histograms for all numeric columns in the dataframe.
    """
    num_cols = len(numeric_cols)
    plt.figure(figsize=(15, num_cols * 3))
    for i, feature in enumerate(numeric_cols, 1):
        plt.subplot(num_cols, 3, i)
        sns.histplot(df[feature], kde=True)
        plt.title(f'Histogram of {feature}')
    plt.tight_layout()
    plt.show()

def visualize_outliers(df, numeric_cols, title_prefix):
    """
    Visualizes outliers in the numeric columns using boxplots.
    """
    cols = 3
    rows = int(np.ceil(len(numeric_cols) / cols))
    plt.figure(figsize=(15, 5 * rows))
    for i, column in enumerate(numeric_cols):
        plt.subplot(rows, cols, i + 1)
        sns.boxplot(y=df[column])
        plt.title(f'{title_prefix} Outliers: {column}')
    plt.tight_layout()
    plt.show()

def remove_outliers_iqr(df, numeric_cols):
    """
    Removes outliers from numeric columns using the IQR method.
    """
    for column in numeric_cols:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Filter rows that fall within the IQR bounds
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df
