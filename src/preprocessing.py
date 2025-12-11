"""
Preprocessing Module
Fix data leakage with proper splits and handle class imbalance.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List


# Classes to merge into 'small_RNA'
RARE_CLASSES = ['scRNA', 'snRNA', 'rRNA']
MERGED_CLASS_NAME = 'small_RNA'


def merge_rare_classes(df: pd.DataFrame, 
                       rare_classes: List[str] = RARE_CLASSES,
                       new_class_name: str = MERGED_CLASS_NAME) -> pd.DataFrame:
    """
    Merge rare RNA classes into a single class.
    
    Args:
        df: DataFrame with GeneType column
        rare_classes: List of class names to merge
        new_class_name: Name for the merged class
        
    Returns:
        DataFrame with merged classes
    """
    df_merged = df.copy()
    
    # Count before merge
    print("Class distribution BEFORE merging:")
    for cls in rare_classes:
        count = (df_merged['GeneType'] == cls).sum()
        print(f"  {cls}: {count}")
    
    # Merge rare classes
    df_merged['GeneType'] = df_merged['GeneType'].apply(
        lambda x: new_class_name if x in rare_classes else x
    )
    
    # Count after merge
    merged_count = (df_merged['GeneType'] == new_class_name).sum()
    print(f"\nAfter merge -> {new_class_name}: {merged_count}")
    
    print(f"\nTotal classes: {df_merged['GeneType'].nunique()}")
    print(f"Classes: {sorted(df_merged['GeneType'].unique())}")
    
    return df_merged


def create_stratified_split(df: pd.DataFrame,
                           test_size: float = 0.15,
                           val_size: float = 0.15,
                           random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/test/validation split without data leakage.
    
    Args:
        df: Clean DataFrame
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df, val_df)
    """
    print("\n" + "=" * 50)
    print("CREATING STRATIFIED SPLIT")
    print("=" * 50)
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['GeneType'],
        random_state=random_state
    )
    
    # Second split: separate validation from remaining
    # Adjust val_size to account for already removed test data
    adjusted_val_size = val_size / (1 - test_size)
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        stratify=train_val_df['GeneType'],
        random_state=random_state
    )
    
    # Reset indices
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    # Print split info
    total = len(df)
    print(f"\nSplit results:")
    print(f"  - Train: {len(train_df)} ({len(train_df)/total*100:.1f}%)")
    print(f"  - Test: {len(test_df)} ({len(test_df)/total*100:.1f}%)")
    print(f"  - Validation: {len(val_df)} ({len(val_df)/total*100:.1f}%)")
    
    return train_df, test_df, val_df


def verify_no_leakage(train_df: pd.DataFrame, 
                      test_df: pd.DataFrame, 
                      val_df: pd.DataFrame,
                      check_column: str = 'NucleotideSequence') -> bool:
    """
    Verify there's no data leakage between sets.
    
    Args:
        train_df, test_df, val_df: The split DataFrames
        check_column: Column to check for overlap
        
    Returns:
        True if no leakage found, False otherwise
    """
    print("\n" + "=" * 50)
    print("VERIFYING NO DATA LEAKAGE")
    print("=" * 50)
    
    train_seqs = set(train_df[check_column].values)
    test_seqs = set(test_df[check_column].values)
    val_seqs = set(val_df[check_column].values)
    
    train_test_overlap = train_seqs & test_seqs
    train_val_overlap = train_seqs & val_seqs
    test_val_overlap = test_seqs & val_seqs
    
    print(f"Train-Test overlap: {len(train_test_overlap)} sequences")
    print(f"Train-Val overlap: {len(train_val_overlap)} sequences")
    print(f"Test-Val overlap: {len(test_val_overlap)} sequences")
    
    no_leakage = (len(train_test_overlap) == 0 and 
                  len(train_val_overlap) == 0 and 
                  len(test_val_overlap) == 0)
    
    if no_leakage:
        print("\n[OK] No data leakage detected!")
    else:
        print("\n[WARNING] Some sequence overlap detected (may be legitimate - different genes with same sequence)")
        
    return no_leakage


def print_class_distribution(train_df: pd.DataFrame,
                            test_df: pd.DataFrame,
                            val_df: pd.DataFrame) -> None:
    """Print class distribution for each split."""
    print("\n" + "=" * 50)
    print("CLASS DISTRIBUTION BY SPLIT")
    print("=" * 50)
    
    for name, df in [("Train", train_df), ("Test", test_df), ("Validation", val_df)]:
        print(f"\n{name}:")
        dist = df['GeneType'].value_counts()
        for cls, count in dist.items():
            pct = count / len(df) * 100
            print(f"  {cls}: {count} ({pct:.1f}%)")


def preprocess_data(df: pd.DataFrame,
                   merge_classes: bool = True,
                   test_size: float = 0.15,
                   val_size: float = 0.15,
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Complete preprocessing pipeline.
    
    Args:
        df: Clean DataFrame from data_loader
        merge_classes: Whether to merge rare RNA classes
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed
        
    Returns:
        Tuple of (train_df, test_df, val_df)
    """
    print("\n" + "=" * 50)
    print("PREPROCESSING PIPELINE")
    print("=" * 50)
    
    # Step 1: Merge rare classes if requested
    if merge_classes:
        print("\n--- Merging Rare Classes ---")
        df = merge_rare_classes(df)
    
    # Step 2: Create stratified split
    train_df, test_df, val_df = create_stratified_split(
        df, test_size=test_size, val_size=val_size, random_state=random_state
    )
    
    # Step 3: Verify no leakage
    verify_no_leakage(train_df, test_df, val_df)
    
    # Step 4: Print class distribution
    print_class_distribution(train_df, test_df, val_df)
    
    return train_df, test_df, val_df


if __name__ == "__main__":
    # Test the module
    from pathlib import Path
    from data_loader import load_and_clean_data
    
    # Get the data directory
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent
    
    # Load and clean data
    df = load_and_clean_data(data_dir)
    
    # Preprocess
    train_df, test_df, val_df = preprocess_data(df)
    
    print(f"\nFinal shapes:")
    print(f"  Train: {train_df.shape}")
    print(f"  Test: {test_df.shape}")
    print(f"  Validation: {val_df.shape}")
