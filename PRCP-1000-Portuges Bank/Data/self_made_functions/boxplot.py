import matplotlib.pyplot as plt
import seaborn as sns

def plot_all_boxplots(data, cols_per_row=3, color='skyblue'):
    """
    Plots boxplots of all numerical columns in the dataset.
    
    Parameters:
        data (DataFrame): Your pandas DataFrame
        cols_per_row (int): Number of plots per row (default 3)
        color (str): Box color (default 'skyblue')
    """
    # Select only numerical columns
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns
    n = len(num_cols)
    
    if n == 0:
        print("No numerical columns found!")
        return
    
    # Calculate rows and columns
    rows = (n + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(5 * cols_per_row, 4 * rows))
    axes = axes.flatten()
    
    # Plot each column
    for i, col in enumerate(num_cols):
        sns.boxplot(y=data[col], ax=axes[i], color=color)
        axes[i].set_title(col, fontsize=12)
    
    # Remove empty subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()
