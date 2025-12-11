"""
Data Loader Module
Load and merge gene classification datasets.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional


def load_csv(filepath: str) -> pd.DataFrame:
    """Load a single CSV file."""
    df = pd.read_csv(filepath)
    return df


def load_all_data(data_dir: str) -> pd.DataFrame:
    """
    Load and merge all CSV files (train, test, validation).
    
    Args:
        data_dir: Path to directory containing CSV files
        
    Returns:
        Merged DataFrame with all data
    """
    data_path = Path(data_dir)
    
    # Load all datasets
    train_df = load_csv(data_path / "train.csv")
    test_df = load_csv(data_path / "test.csv")
    validation_df = load_csv(data_path / "validation.csv")
    
    # Add source column to track origin (optional, for debugging)
    train_df['_source'] = 'train'
    test_df['_source'] = 'test'
    validation_df['_source'] = 'validation'
    
    # Merge all data
    all_data = pd.concat([train_df, test_df, validation_df], ignore_index=True)
    
    print(f"Loaded data:")
    print(f"  - Train: {len(train_df)} rows")
    print(f"  - Test: {len(test_df)} rows")
    print(f"  - Validation: {len(validation_df)} rows")
    print(f"  - Total: {len(all_data)} rows")
    
    return all_data


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by removing unnecessary columns and duplicates.
    
    Args:
        df: Raw merged DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    # Make a copy to avoid modifying original
    df_clean = df.copy()
    
    # Identify columns to drop
    columns_to_drop = []
    
    # Drop unnamed index column if exists
    unnamed_cols = [col for col in df_clean.columns if 'Unnamed' in str(col)]
    columns_to_drop.extend(unnamed_cols)
    
    # Drop GeneGroupMethod (constant value - "NCBI Ortholog")
    if 'GeneGroupMethod' in df_clean.columns:
        columns_to_drop.append('GeneGroupMethod')
    
    # Remove duplicates by NCBIGeneID BEFORE dropping it
    original_count = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['NCBIGeneID'], keep='first')
    duplicates_removed = original_count - len(df_clean)
    
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate rows (by NCBIGeneID)")
    
    # Now drop NCBIGeneID (unique identifier, not useful for prediction)
    if 'NCBIGeneID' in df_clean.columns:
        columns_to_drop.append('NCBIGeneID')
    
    # Drop the _source column (was only for debugging)
    if '_source' in df_clean.columns:
        columns_to_drop.append('_source')
    
    # Drop all identified columns
    df_clean = df_clean.drop(columns=columns_to_drop, errors='ignore')
    
    print(f"Dropped columns: {columns_to_drop}")
    print(f"Remaining columns: {list(df_clean.columns)}")
    print(f"Final dataset size: {len(df_clean)} rows")
    
    return df_clean


def load_and_clean_data(data_dir: str) -> pd.DataFrame:
    """
    Complete data loading pipeline.
    
    Args:
        data_dir: Path to directory containing CSV files
        
    Returns:
        Clean, merged DataFrame ready for preprocessing
    """
    print("=" * 50)
    print("LOADING DATA")
    print("=" * 50)
    
    # Load all data
    all_data = load_all_data(data_dir)
    
    print("\n" + "=" * 50)
    print("CLEANING DATA")
    print("=" * 50)
    
    # Clean data
    clean_df = clean_data(all_data)
    
    print("\n" + "=" * 50)
    print("DATA SUMMARY")
    print("=" * 50)
    print(f"\nGeneType distribution:")
    print(clean_df['GeneType'].value_counts())
    
    return clean_df


if __name__ == "__main__":
    # Test the module
    import os
    
    # Get the data directory (parent of src)
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent
    
    df = load_and_clean_data(data_dir)
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
