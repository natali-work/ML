"""
Feature Engineering Module
Extract features from nucleotide sequences for ML classification.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from itertools import product
from collections import Counter


# All possible 3-mers (k=3)
NUCLEOTIDES = ['A', 'C', 'G', 'T']
K_MER_SIZE = 3
ALL_KMERS = [''.join(p) for p in product(NUCLEOTIDES, repeat=K_MER_SIZE)]


def clean_sequence(seq: str) -> str:
    """Remove angle brackets and convert U to T."""
    seq = seq.strip('<>')
    seq = seq.upper()
    seq = seq.replace('U', 'T')  # Convert RNA to DNA notation
    return seq


def get_sequence_length(seq: str) -> int:
    """Get the length of a cleaned sequence."""
    return len(clean_sequence(seq))


def get_gc_content(seq: str) -> float:
    """Calculate GC content (proportion of G and C)."""
    seq = clean_sequence(seq)
    if len(seq) == 0:
        return 0.0
    gc_count = seq.count('G') + seq.count('C')
    return gc_count / len(seq)


def get_nucleotide_frequencies(seq: str) -> Dict[str, float]:
    """Calculate frequency of each nucleotide."""
    seq = clean_sequence(seq)
    length = len(seq)
    
    if length == 0:
        return {'a_freq': 0, 'c_freq': 0, 'g_freq': 0, 't_freq': 0}
    
    return {
        'a_freq': seq.count('A') / length,
        'c_freq': seq.count('C') / length,
        'g_freq': seq.count('G') / length,
        't_freq': seq.count('T') / length
    }


def get_at_gc_ratio(seq: str) -> float:
    """Calculate AT/GC ratio."""
    seq = clean_sequence(seq)
    at_count = seq.count('A') + seq.count('T')
    gc_count = seq.count('G') + seq.count('C')
    
    if gc_count == 0:
        return 0.0  # Avoid division by zero
    
    return at_count / gc_count


def get_kmer_frequencies(seq: str, k: int = K_MER_SIZE) -> Dict[str, float]:
    """
    Calculate k-mer frequencies.
    
    Args:
        seq: Nucleotide sequence
        k: Size of k-mers (default 3)
        
    Returns:
        Dictionary of k-mer frequencies
    """
    seq = clean_sequence(seq)
    
    # Count k-mers
    kmer_counts = Counter()
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        # Only count valid k-mers (all nucleotides are A, C, G, T)
        if all(c in NUCLEOTIDES for c in kmer):
            kmer_counts[kmer] += 1
    
    # Convert to frequencies
    total_kmers = sum(kmer_counts.values())
    
    if total_kmers == 0:
        return {f'kmer_{kmer}': 0.0 for kmer in ALL_KMERS}
    
    return {f'kmer_{kmer}': kmer_counts.get(kmer, 0) / total_kmers 
            for kmer in ALL_KMERS}


def get_symbol_features(symbol: str) -> Dict[str, int]:
    """
    Extract features from gene symbol.
    
    Args:
        symbol: Gene symbol string
        
    Returns:
        Dictionary of binary features
    """
    symbol = str(symbol).upper()
    
    return {
        'ends_with_p': 1 if symbol.endswith('P') else 0,
        'starts_with_mir': 1 if symbol.startswith('MIR') else 0,
        'starts_with_trna': 1 if symbol.startswith('TRNA') else 0,
        'starts_with_loc': 1 if symbol.startswith('LOC') else 0,
        'starts_with_rnu': 1 if symbol.startswith('RNU') else 0,
        'starts_with_snor': 1 if symbol.startswith('SNOR') else 0,
    }


def extract_features_single(row: pd.Series) -> Dict:
    """
    Extract all features from a single row.
    
    Args:
        row: DataFrame row with NucleotideSequence and Symbol columns
        
    Returns:
        Dictionary of all features
    """
    seq = row['NucleotideSequence']
    symbol = row.get('Symbol', '')
    
    features = {}
    
    # Sequence length
    features['seq_length'] = get_sequence_length(seq)
    
    # GC content
    features['gc_content'] = get_gc_content(seq)
    
    # Nucleotide frequencies
    features.update(get_nucleotide_frequencies(seq))
    
    # AT/GC ratio
    features['at_gc_ratio'] = get_at_gc_ratio(seq)
    
    # K-mer frequencies (64 features for k=3)
    features.update(get_kmer_frequencies(seq))
    
    # Symbol features
    if symbol:
        features.update(get_symbol_features(symbol))
    
    return features


def extract_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Extract all features from a DataFrame.
    
    Args:
        df: DataFrame with NucleotideSequence column
        verbose: Whether to print progress
        
    Returns:
        DataFrame with extracted features
    """
    if verbose:
        print(f"Extracting features from {len(df)} sequences...")
    
    # Extract features for all rows
    features_list = []
    
    for idx, row in df.iterrows():
        features = extract_features_single(row)
        features_list.append(features)
        
        if verbose and (idx + 1) % 5000 == 0:
            print(f"  Processed {idx + 1}/{len(df)} sequences")
    
    # Create features DataFrame
    features_df = pd.DataFrame(features_list)
    
    if verbose:
        print(f"Extracted {len(features_df.columns)} features")
        print(f"Features: {list(features_df.columns)[:10]}... (and {len(features_df.columns)-10} more)")
    
    return features_df


def prepare_ml_data(train_df: pd.DataFrame,
                   test_df: pd.DataFrame,
                   val_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, 
                                                   np.ndarray, np.ndarray,
                                                   np.ndarray, np.ndarray,
                                                   List[str], List[str]]:
    """
    Prepare data for ML training.
    
    Args:
        train_df, test_df, val_df: DataFrames from preprocessing
        
    Returns:
        X_train, X_test, X_val, y_train, y_test, y_val, feature_names, class_names
    """
    print("\n" + "=" * 50)
    print("PREPARING ML DATA")
    print("=" * 50)
    
    # Extract features
    print("\nExtracting training features...")
    X_train_df = extract_features(train_df)
    
    print("\nExtracting test features...")
    X_test_df = extract_features(test_df)
    
    print("\nExtracting validation features...")
    X_val_df = extract_features(val_df)
    
    # Get labels
    y_train = train_df['GeneType'].values
    y_test = test_df['GeneType'].values
    y_val = val_df['GeneType'].values
    
    # Convert to numpy arrays
    X_train = X_train_df.values
    X_test = X_test_df.values
    X_val = X_val_df.values
    
    feature_names = list(X_train_df.columns)
    class_names = sorted(train_df['GeneType'].unique())
    
    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"\nNumber of features: {len(feature_names)}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    
    return X_train, X_test, X_val, y_train, y_test, y_val, feature_names, class_names


if __name__ == "__main__":
    # Test the module
    from pathlib import Path
    from data_loader import load_and_clean_data
    from preprocessing import preprocess_data
    
    # Get the data directory
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent
    
    # Load, clean, and preprocess
    df = load_and_clean_data(data_dir)
    train_df, test_df, val_df = preprocess_data(df)
    
    # Extract features
    X_train, X_test, X_val, y_train, y_test, y_val, feature_names, class_names = \
        prepare_ml_data(train_df, test_df, val_df)
    
    print(f"\nFeature names (first 10): {feature_names[:10]}")
    print(f"Sample X_train row: {X_train[0][:5]}...")
