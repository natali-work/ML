"""
Gene Classification ML Pipeline

Modules:
- data_loader: Load and merge CSV datasets
- preprocessing: Fix data leakage, merge rare classes
- features: Extract sequence features
- model: Train and evaluate classifiers
- main: Run complete pipeline
"""

from .data_loader import load_and_clean_data
from .preprocessing import preprocess_data
from .features import prepare_ml_data, extract_features
from .model import GeneClassifier, train_and_evaluate

__all__ = [
    'load_and_clean_data',
    'preprocess_data', 
    'prepare_ml_data',
    'extract_features',
    'GeneClassifier',
    'train_and_evaluate'
]
