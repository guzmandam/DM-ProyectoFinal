"""
Module for retrieving and saving brewery sales data from the 'ankurnapa/Brewery_sales' dataset.

This module provides functionality to load the dataset from Hugging Face
and save it in various formats.
"""

import os
from datasets import load_dataset

DATASET_REF = "ankurnapa/Brewery_sales"

class Retriever:
    """
    A class to retrieve and manage brewery sales data.
    
    This class handles loading data from a predefined dataset and 
    provides methods to save the data in different formats.
    """

    def __init__(self):
        """
        Initialize the Retriever and load the brewery sales dataset.
        """
        self.dataset = load_dataset(DATASET_REF, split="train")

    def load_data(self):
        """
        Return the loaded dataset.
        
        Returns:
            Dataset: The loaded brewery sales dataset.
        """
        return self.dataset

    def save_data(self, file_type="parquet"):
        """
        Save the dataset to a file in the specified format.
        
        Args:
            file_type (str, optional): Format to save the data. 
                Supported formats: 'parquet', 'csv', 'json'. Defaults to "parquet".
        
        Returns:
            str: Path to the saved file.
            
        Raises:
            ValueError: If an unsupported file type is specified.
        """
        if not os.path.exists("./data"):
            os.makedirs("./data")

        output_file = f"./data/brewery_sales.{file_type}"

        if file_type == "parquet":
            self.dataset.to_parquet(output_file)
        elif file_type == "csv":
            self.dataset.to_csv(output_file, index=False)
        elif file_type == "json":
            self.dataset.to_json(output_file, orient="records", lines=True)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        print(f"Datos guardados en {output_file}")

        return output_file

if __name__ == "__main__":
    retriever = Retriever()
    retriever.save_data(file_type="parquet")