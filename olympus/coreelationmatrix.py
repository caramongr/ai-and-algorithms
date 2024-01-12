import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Function to create and plot a correlation matrix
def create_correlation_matrix(file_path, excluded_columns=None):
    """
    Creates and plots a correlation matrix for the specified Excel file.
    
    Args:
    file_path (str): Path to the Excel file.
    excluded_columns (list of str): List of column names to be excluded.
    """
    # Load the data
    data = pd.read_excel('MasterDataFile.xlsx')

    # Exclude specified columns
    if excluded_columns:
        data.drop(columns=excluded_columns, inplace=True, errors='ignore')

    # Calculate the correlation matrix
    correlation_matrix = data.corr()

    # Plot the correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

# Specify the file path
file_path = 'path_to_your_excel_file.xlsx'  # Replace with your file path

# Enter the names of columns to exclude, e.g., ['Column1', 'Column2']
excluded_columns = ['Column_to_exclude']  # Replace with your column names

# Create and plot the correlation matrix
create_correlation_matrix(file_path, excluded_columns=excluded_columns)
