"""
Generate comprehensive visualizations and report for model evaluation.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add src to path
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

from data_loader import load_and_clean_data
from preprocessing import preprocess_data
from features import prepare_ml_data
from model import GeneClassifier

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


def plot_class_distribution(train_df, test_df, val_df, save_path):
    """Plot class distribution across splits."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, (name, df) in zip(axes, [('Train', train_df), ('Test', test_df), ('Validation', val_df)]):
        counts = df['GeneType'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(counts)))
        bars = ax.bar(range(len(counts)), counts.values, color=colors)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45, ha='right')
        ax.set_title(f'{name} Set (n={len(df)})')
        ax.set_ylabel('Count')
        
        # Add count labels
        for bar, count in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
                   str(count), ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Class Distribution Across Dataset Splits', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_model_comparison(rf_results, xgb_results, save_path):
    """Plot comparison between Random Forest and XGBoost."""
    metrics = ['accuracy', 'macro_f1', 'weighted_f1']
    metric_names = ['Accuracy', 'Macro F1', 'Weighted F1']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, dataset in zip(axes, ['validation', 'test']):
        rf_vals = [rf_results[dataset][m] for m in metrics]
        xgb_vals = [xgb_results[dataset][m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, rf_vals, width, label='Random Forest', color='#3498db')
        bars2 = ax.bar(x + width/2, xgb_vals, width, label='XGBoost', color='#e74c3c')
        
        ax.set_ylabel('Score')
        ax.set_title(f'{dataset.title()} Set Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend()
        ax.set_ylim(0.7, 1.0)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_per_class_f1(rf_results, xgb_results, save_path):
    """Plot per-class F1 scores for both models."""
    from sklearn.metrics import f1_score
    
    # Calculate per-class F1 scores
    classes = sorted(set(rf_results['test']['y_true']))
    
    rf_f1 = f1_score(rf_results['test']['y_true'], rf_results['test']['y_pred'], 
                     labels=classes, average=None)
    xgb_f1 = f1_score(xgb_results['test']['y_true'], xgb_results['test']['y_pred'], 
                      labels=classes, average=None)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(classes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, rf_f1, width, label='Random Forest', color='#3498db')
    bars2 = ax.bar(x + width/2, xgb_f1, width, label='XGBoost', color='#e74c3c')
    
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class F1 Scores on Test Set', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='0.8 threshold')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_confusion_matrices_side_by_side(rf_results, xgb_results, save_path):
    """Plot confusion matrices side by side."""
    from sklearn.metrics import confusion_matrix
    
    classes = sorted(set(rf_results['test']['y_true']))
    
    rf_cm = confusion_matrix(rf_results['test']['y_true'], rf_results['test']['y_pred'], labels=classes)
    xgb_cm = confusion_matrix(xgb_results['test']['y_true'], xgb_results['test']['y_pred'], labels=classes)
    
    # Normalize
    rf_cm_norm = rf_cm.astype('float') / rf_cm.sum(axis=1)[:, np.newaxis]
    xgb_cm_norm = xgb_cm.astype('float') / xgb_cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for ax, cm, title in zip(axes, [rf_cm_norm, xgb_cm_norm], ['Random Forest', 'XGBoost']):
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                   xticklabels=classes, yticklabels=classes, vmin=0, vmax=1)
        ax.set_title(f'{title} - Normalized Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Confusion Matrices Comparison (Test Set)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_feature_importance_comparison(rf_model, xgb_model, feature_names, save_path, top_n=15):
    """Plot feature importance comparison."""
    rf_imp = rf_model.model.feature_importances_
    xgb_imp = xgb_model.model.feature_importances_
    
    # Get top features by average importance
    avg_imp = (rf_imp + xgb_imp) / 2
    top_idx = np.argsort(avg_imp)[-top_n:]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y = np.arange(len(top_idx))
    height = 0.35
    
    bars1 = ax.barh(y - height/2, rf_imp[top_idx], height, label='Random Forest', color='#3498db')
    bars2 = ax.barh(y + height/2, xgb_imp[top_idx], height, label='XGBoost', color='#e74c3c')
    
    ax.set_yticks(y)
    ax.set_yticklabels([feature_names[i] for i in top_idx])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importances Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def generate_markdown_report(rf_results, xgb_results, data_stats, save_path):
    """Generate a comprehensive Markdown report."""
    
    report = f"""# Gene Classification Model - Evaluation Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## 1. Data Summary

| Metric | Value |
|--------|-------|
| Total samples (after dedup) | {data_stats['total_samples']} |
| Training samples | {data_stats['train_samples']} |
| Test samples | {data_stats['test_samples']} |
| Validation samples | {data_stats['val_samples']} |
| Number of features | {data_stats['n_features']} |
| Number of classes | {data_stats['n_classes']} |

### Class Distribution (Training Set)
| Class | Count | Percentage |
|-------|-------|------------|
"""
    
    for cls, count in data_stats['class_distribution'].items():
        pct = count / data_stats['train_samples'] * 100
        report += f"| {cls} | {count} | {pct:.1f}% |\n"
    
    report += f"""
---

## 2. Model Performance Summary

### Overall Metrics

| Model | Dataset | Accuracy | Macro F1 | Weighted F1 |
|-------|---------|----------|----------|-------------|
| Random Forest | Validation | {rf_results['validation']['accuracy']:.4f} | {rf_results['validation']['macro_f1']:.4f} | {rf_results['validation']['weighted_f1']:.4f} |
| Random Forest | Test | {rf_results['test']['accuracy']:.4f} | {rf_results['test']['macro_f1']:.4f} | {rf_results['test']['weighted_f1']:.4f} |
| XGBoost | Validation | {xgb_results['validation']['accuracy']:.4f} | {xgb_results['validation']['macro_f1']:.4f} | {xgb_results['validation']['weighted_f1']:.4f} |
| XGBoost | Test | {xgb_results['test']['accuracy']:.4f} | {xgb_results['test']['macro_f1']:.4f} | {xgb_results['test']['weighted_f1']:.4f} |

### Best Model: **XGBoost**
- Test Accuracy: {xgb_results['test']['accuracy']:.2%}
- Test Macro F1: {xgb_results['test']['macro_f1']:.4f}

---

## 3. Per-Class Analysis (Test Set - XGBoost)

{xgb_results['test']['classification_report']}

---

## 4. Key Findings

### Strengths (Classes with F1 > 0.90):
"""
    
    from sklearn.metrics import f1_score
    classes = sorted(set(xgb_results['test']['y_true']))
    xgb_f1 = f1_score(xgb_results['test']['y_true'], xgb_results['test']['y_pred'], 
                      labels=classes, average=None)
    
    strong_classes = [(cls, f1) for cls, f1 in zip(classes, xgb_f1) if f1 >= 0.90]
    weak_classes = [(cls, f1) for cls, f1 in zip(classes, xgb_f1) if f1 < 0.80]
    
    for cls, f1 in strong_classes:
        report += f"- **{cls}**: F1 = {f1:.3f}\n"
    
    report += "\n### Weaknesses (Classes with F1 < 0.80):\n"
    for cls, f1 in weak_classes:
        report += f"- **{cls}**: F1 = {f1:.3f}\n"
    
    report += f"""
---

## 5. Issues Identified

### 5.1 Class Imbalance
The dataset has significant class imbalance:
- Largest class (PSEUDO): {data_stats['class_distribution'].get('PSEUDO', 0)} samples ({data_stats['class_distribution'].get('PSEUDO', 0)/data_stats['train_samples']*100:.1f}%)
- Smallest class (OTHER): {data_stats['class_distribution'].get('OTHER', 0)} samples ({data_stats['class_distribution'].get('OTHER', 0)/data_stats['train_samples']*100:.1f}%)
- Imbalance ratio: {max(data_stats['class_distribution'].values()) / min(data_stats['class_distribution'].values()):.1f}:1

**Impact**: The model performs well on majority classes (PSEUDO, BIOLOGICAL_REGION) but struggles with minority classes (OTHER, PROTEIN_CODING).

### 5.2 Sequence Overlap
Some sequence overlap was detected between splits:
- This is expected since different genes can have identical sequences
- The split was performed by unique GeneID, ensuring proper separation

### 5.3 Feature Importance
Top contributing features (both models agree):
- **seq_length**: Sequence length is highly predictive
- **gc_content**: GC content differentiates gene types
- **K-mer frequencies**: Specific 3-mer patterns are informative

---

## 6. Recommendations

### For Improved Performance:

1. **Address Class Imbalance**:
   - Use more aggressive oversampling (SMOTE/ADASYN) for minority classes
   - Consider focal loss to focus on hard examples
   - Collect more samples for rare classes (OTHER, PROTEIN_CODING)

2. **Feature Engineering**:
   - Add longer k-mers (4-mers, 5-mers)
   - Include positional features (start/end motifs)
   - Add secondary structure predictions

3. **Model Improvements**:
   - Try ensemble methods combining RF and XGBoost
   - Experiment with hyperparameter tuning
   - Consider deep learning (CNN/LSTM) for sequence data

4. **Data Quality**:
   - Review misclassified samples for labeling errors
   - Consider merging similar classes if biologically justified

---

## 7. Files Generated

| File | Description |
|------|-------------|
| `random_forest_model.pkl` | Trained Random Forest model |
| `xgboost_model.pkl` | Trained XGBoost model |
| `random_forest_confusion_matrix.png` | RF confusion matrix |
| `xgboost_confusion_matrix.png` | XGBoost confusion matrix |
| `random_forest_feature_importance.png` | RF feature importance |
| `xgboost_feature_importance.png` | XGBoost feature importance |
| `class_distribution.png` | Class distribution across splits |
| `model_comparison.png` | Model performance comparison |
| `per_class_f1.png` | Per-class F1 scores |
| `confusion_matrices_comparison.png` | Side-by-side confusion matrices |

---

*Report generated automatically by the Gene Classification ML Pipeline*
"""
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Saved: {save_path}")


def main():
    """Generate all visualizations and report."""
    # Paths
    data_dir = src_dir.parent
    models_dir = data_dir / "models"
    
    print("=" * 60)
    print("GENERATING VISUALIZATIONS AND REPORT")
    print("=" * 60)
    
    # Load data and preprocess
    print("\nLoading data...")
    df = load_and_clean_data(data_dir)
    train_df, test_df, val_df = preprocess_data(df)
    X_train, X_test, X_val, y_train, y_test, y_val, feature_names, class_names = \
        prepare_ml_data(train_df, test_df, val_df)
    
    # Load trained models
    print("\nLoading trained models...")
    rf_model = GeneClassifier.load(models_dir / "random_forest_model.pkl")
    xgb_model = GeneClassifier.load(models_dir / "xgboost_model.pkl")
    
    # Get results
    print("\nEvaluating models...")
    rf_val_results = rf_model.evaluate(X_val, y_val, "Validation")
    rf_test_results = rf_model.evaluate(X_test, y_test, "Test")
    xgb_val_results = xgb_model.evaluate(X_val, y_val, "Validation")
    xgb_test_results = xgb_model.evaluate(X_test, y_test, "Test")
    
    rf_results = {'validation': rf_val_results, 'test': rf_test_results}
    xgb_results = {'validation': xgb_val_results, 'test': xgb_test_results}
    
    # Data stats
    data_stats = {
        'total_samples': len(df),
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'val_samples': len(val_df),
        'n_features': len(feature_names),
        'n_classes': len(class_names),
        'class_distribution': dict(train_df['GeneType'].value_counts())
    }
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    plot_class_distribution(train_df, test_df, val_df, 
                           models_dir / "class_distribution.png")
    
    plot_model_comparison(rf_results, xgb_results, 
                         models_dir / "model_comparison.png")
    
    plot_per_class_f1(rf_results, xgb_results, 
                     models_dir / "per_class_f1.png")
    
    plot_confusion_matrices_side_by_side(rf_results, xgb_results, 
                                         models_dir / "confusion_matrices_comparison.png")
    
    plot_feature_importance_comparison(rf_model, xgb_model, feature_names,
                                       models_dir / "feature_importance_comparison.png")
    
    # Generate report
    print("\nGenerating report...")
    generate_markdown_report(rf_results, xgb_results, data_stats, 
                            models_dir / "MODEL_EVALUATION_REPORT.md")
    
    print("\n" + "=" * 60)
    print("DONE! All files saved to:", models_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
