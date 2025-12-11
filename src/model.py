"""
Model Module
Train and evaluate Random Forest and XGBoost classifiers.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Only Random Forest will be available.")

import matplotlib.pyplot as plt
import seaborn as sns


class GeneClassifier:
    """Wrapper class for gene type classification models."""
    
    def __init__(self, model_type: str = 'random_forest', random_state: int = 42):
        """
        Initialize classifier.
        
        Args:
            model_type: 'random_forest' or 'xgboost'
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.label_encoder = LabelEncoder()
        self.class_weights = None
        self.feature_names = None
        self.class_names = None
        
    def _compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Compute balanced class weights."""
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(range(len(classes)), weights))
    
    def _create_model(self, class_weights: Optional[Dict] = None):
        """Create the ML model."""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1,
                verbose=1
            )
        elif self.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost is not installed. Please install it with: pip install xgboost")
            
            # For XGBoost, we need sample weights instead of class_weight parameter
            self.model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='mlogloss',
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            feature_names: List[str] = None, class_names: List[str] = None) -> 'GeneClassifier':
        """
        Train the classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            feature_names: List of feature names
            class_names: List of class names
            
        Returns:
            self
        """
        print(f"\n{'='*50}")
        print(f"TRAINING {self.model_type.upper()}")
        print(f"{'='*50}")
        
        self.feature_names = feature_names
        self.class_names = class_names
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_train)
        
        # Compute class weights
        self.class_weights = self._compute_class_weights(y_encoded)
        print(f"\nClass weights: {self.class_weights}")
        
        # Create model
        self._create_model(self.class_weights)
        
        # Train
        print(f"\nTraining on {len(X_train)} samples...")
        
        if self.model_type == 'xgboost':
            # Compute sample weights for XGBoost
            sample_weights = np.array([self.class_weights[y] for y in y_encoded])
            self.model.fit(X_train, y_encoded, sample_weight=sample_weights)
        else:
            self.model.fit(X_train, y_encoded)
        
        print("Training complete!")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        y_pred_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray, 
                 dataset_name: str = "Test") -> Dict[str, Any]:
        """
        Evaluate the model.
        
        Args:
            X: Features
            y_true: True labels
            dataset_name: Name of dataset for printing
            
        Returns:
            Dictionary of metrics
        """
        print(f"\n{'='*50}")
        print(f"EVALUATING ON {dataset_name.upper()} SET")
        print(f"{'='*50}")
        
        # Predict
        y_pred = self.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Macro F1: {macro_f1:.4f}")
        print(f"  Weighted F1: {weighted_f1:.4f}")
        
        # Classification report
        print(f"\nClassification Report:")
        report = classification_report(y_true, y_pred)
        print(report)
        
        # Store results
        results = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'y_true': y_true,
            'y_pred': y_pred,
            'classification_report': report
        }
        
        return results
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                              save_path: Optional[str] = None) -> None:
        """Plot confusion matrix."""
        # Get unique labels in order
        labels = sorted(set(y_true) | set(y_pred))
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {self.model_type}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.model_type == 'random_forest':
            importances = self.model.feature_importances_
        elif self.model_type == 'xgboost':
            importances = self.model.feature_importances_
        else:
            return None
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names if self.feature_names else range(len(importances)),
            'importance': importances
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def plot_feature_importance(self, top_n: int = 20, 
                               save_path: Optional[str] = None) -> None:
        """Plot feature importance."""
        importance_df = self.get_feature_importance(top_n)
        
        if importance_df is None:
            print("Feature importance not available for this model")
            return
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importances - {self.model_type}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Feature importance plot saved to {save_path}")
        
        plt.close()
    
    def save(self, filepath: str) -> None:
        """Save model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'label_encoder': self.label_encoder,
                'model_type': self.model_type,
                'feature_names': self.feature_names,
                'class_names': self.class_names,
                'class_weights': self.class_weights
            }, f)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'GeneClassifier':
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        classifier = cls(model_type=data['model_type'])
        classifier.model = data['model']
        classifier.label_encoder = data['label_encoder']
        classifier.feature_names = data['feature_names']
        classifier.class_names = data['class_names']
        classifier.class_weights = data['class_weights']
        
        print(f"Model loaded from {filepath}")
        return classifier


def train_and_evaluate(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       feature_names: List[str],
                       class_names: List[str],
                       model_type: str = 'random_forest',
                       save_dir: str = None) -> Tuple[GeneClassifier, Dict]:
    """
    Train and evaluate a model.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        X_val, y_val: Validation data
        feature_names: List of feature names
        class_names: List of class names
        model_type: 'random_forest' or 'xgboost'
        save_dir: Directory to save model and plots
        
    Returns:
        Trained classifier and results dictionary
    """
    # Create and train classifier
    classifier = GeneClassifier(model_type=model_type)
    classifier.fit(X_train, y_train, feature_names, class_names)
    
    # Evaluate on validation set
    val_results = classifier.evaluate(X_val, y_val, "Validation")
    
    # Evaluate on test set
    test_results = classifier.evaluate(X_test, y_test, "Test")
    
    # Save model and plots
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Save model
        classifier.save(save_path / f"{model_type}_model.pkl")
        
        # Plot confusion matrix for test set
        classifier.plot_confusion_matrix(
            test_results['y_true'], 
            test_results['y_pred'],
            save_path / f"{model_type}_confusion_matrix.png"
        )
        
        # Plot feature importance
        classifier.plot_feature_importance(
            top_n=20,
            save_path=save_path / f"{model_type}_feature_importance.png"
        )
    
    results = {
        'validation': val_results,
        'test': test_results
    }
    
    return classifier, results


def compare_models(results_rf: Dict, results_xgb: Dict) -> None:
    """Compare Random Forest and XGBoost results."""
    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)
    
    print("\n{:<20} {:<15} {:<15}".format("Metric", "Random Forest", "XGBoost"))
    print("-" * 50)
    
    for dataset in ['validation', 'test']:
        print(f"\n{dataset.upper()} SET:")
        for metric in ['accuracy', 'macro_f1', 'weighted_f1']:
            rf_val = results_rf[dataset][metric]
            xgb_val = results_xgb[dataset][metric]
            winner = "RF" if rf_val > xgb_val else "XGB" if xgb_val > rf_val else "TIE"
            print(f"  {metric:<18} {rf_val:.4f}          {xgb_val:.4f}         ({winner})")


if __name__ == "__main__":
    # Test the module
    from pathlib import Path
    from data_loader import load_and_clean_data
    from preprocessing import preprocess_data
    from features import prepare_ml_data
    
    # Get the data directory
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent
    models_dir = data_dir / "models"
    
    # Load, clean, preprocess, and extract features
    df = load_and_clean_data(data_dir)
    train_df, test_df, val_df = preprocess_data(df)
    X_train, X_test, X_val, y_train, y_test, y_val, feature_names, class_names = \
        prepare_ml_data(train_df, test_df, val_df)
    
    # Train Random Forest
    rf_classifier, rf_results = train_and_evaluate(
        X_train, y_train, X_test, y_test, X_val, y_val,
        feature_names, class_names,
        model_type='random_forest',
        save_dir=models_dir
    )
    
    # Train XGBoost (if available)
    if XGBOOST_AVAILABLE:
        xgb_classifier, xgb_results = train_and_evaluate(
            X_train, y_train, X_test, y_test, X_val, y_val,
            feature_names, class_names,
            model_type='xgboost',
            save_dir=models_dir
        )
        
        # Compare models
        compare_models(rf_results, xgb_results)
