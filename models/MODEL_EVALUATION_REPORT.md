# Gene Classification Model - Evaluation Report

Generated: 2025-12-11 15:14

---

## 1. Data Summary

| Metric | Value |
|--------|-------|
| Total samples (after dedup) | 23779 |
| Training samples | 16645 |
| Test samples | 3567 |
| Validation samples | 3567 |
| Number of features | 77 |
| Number of classes | 8 |

### Class Distribution (Training Set)
| Class | Count | Percentage |
|-------|-------|------------|
| PSEUDO | 7466 | 44.9% |
| BIOLOGICAL_REGION | 5065 | 30.4% |
| ncRNA | 1879 | 11.3% |
| snoRNA | 838 | 5.0% |
| PROTEIN_CODING | 396 | 2.4% |
| tRNA | 383 | 2.3% |
| small_RNA | 356 | 2.1% |
| OTHER | 262 | 1.6% |

---

## 2. Model Performance Summary

### Overall Metrics

| Model | Dataset | Accuracy | Macro F1 | Weighted F1 |
|-------|---------|----------|----------|-------------|
| Random Forest | Validation | 0.9148 | 0.8522 | 0.9120 |
| Random Forest | Test | 0.9176 | 0.8626 | 0.9149 |
| XGBoost | Validation | 0.9333 | 0.8944 | 0.9327 |
| XGBoost | Test | 0.9280 | 0.8824 | 0.9273 |

### Best Model: **XGBoost**
- Test Accuracy: 92.80%
- Test Macro F1: 0.8824

---

## 3. Per-Class Analysis (Test Set - XGBoost)

                   precision    recall  f1-score   support

BIOLOGICAL_REGION       0.94      0.97      0.95      1085
            OTHER       0.77      0.82      0.79        56
   PROTEIN_CODING       0.81      0.69      0.75        85
           PSEUDO       0.95      0.94      0.95      1600
            ncRNA       0.90      0.82      0.86       403
        small_RNA       0.94      0.86      0.90        76
           snoRNA       0.85      0.97      0.91       180
             tRNA       0.94      0.99      0.96        82

         accuracy                           0.93      3567
        macro avg       0.89      0.88      0.88      3567
     weighted avg       0.93      0.93      0.93      3567


---

## 4. Key Findings

### Strengths (Classes with F1 > 0.90):
- **BIOLOGICAL_REGION**: F1 = 0.951
- **PSEUDO**: F1 = 0.945
- **snoRNA**: F1 = 0.906
- **tRNA**: F1 = 0.964

### Weaknesses (Classes with F1 < 0.80):
- **OTHER**: F1 = 0.793
- **PROTEIN_CODING**: F1 = 0.747

---

## 5. Issues Identified

### 5.1 Class Imbalance
The dataset has significant class imbalance:
- Largest class (PSEUDO): 7466 samples (44.9%)
- Smallest class (OTHER): 262 samples (1.6%)
- Imbalance ratio: 28.5:1

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
