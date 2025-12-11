# -*- coding: utf-8 -*-
"""
Script for analyzing data issues - ML Dataset
"""

import pandas as pd
import numpy as np
from collections import Counter

# Load data
print("=" * 60)
print("Loading data...")
print("=" * 60)

train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)
validation = pd.read_csv('validation.csv', index_col=0)

print(f"\nTrain size: {len(train)}")
print(f"Test size: {len(test)}")
print(f"Validation size: {len(validation)}")

# ============================================
# Issue 1: Missing Values
# ============================================
print("\n" + "=" * 60)
print("Issue 1: Missing Values")
print("=" * 60)

for name, df in [('Train', train), ('Test', test), ('Validation', validation)]:
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n{name}:")
        print(missing[missing > 0])
    else:
        print(f"\n{name}: No missing values [OK]")

# ============================================
# Issue 2: Class Imbalance
# ============================================
print("\n" + "=" * 60)
print("Issue 2: Class Imbalance")
print("=" * 60)

train_dist = train['GeneType'].value_counts()
print("\nDistribution in Train:")
for gene_type, count in train_dist.items():
    pct = count / len(train) * 100
    print(f"  {gene_type}: {count} ({pct:.2f}%)")

# Imbalance ratio
max_class = train_dist.max()
min_class = train_dist.min()
imbalance_ratio = max_class / min_class
print(f"\nImbalance ratio (max/min): {imbalance_ratio:.0f}:1")

# ============================================
# Issue 3: Very Rare Classes
# ============================================
print("\n" + "=" * 60)
print("Issue 3: Very Rare Classes")
print("=" * 60)

rare_threshold = 50
for name, df in [('Train', train), ('Test', test), ('Validation', validation)]:
    rare_classes = df['GeneType'].value_counts()
    rare = rare_classes[rare_classes < rare_threshold]
    if len(rare) > 0:
        print(f"\n{name} - Classes with less than {rare_threshold} samples:")
        for cls, cnt in rare.items():
            print(f"  {cls}: {cnt}")

# Check for missing classes
print("\n\nMissing classes in each set:")
all_classes = set(train['GeneType'].unique())
for name, df in [('Test', test), ('Validation', validation)]:
    df_classes = set(df['GeneType'].unique())
    missing_classes = all_classes - df_classes
    if missing_classes:
        print(f"  {name}: MISSING {missing_classes}")
    else:
        print(f"  {name}: All classes present [OK]")

# ============================================
# Issue 4: Duplicates
# ============================================
print("\n" + "=" * 60)
print("Issue 4: Duplicates")
print("=" * 60)

for name, df in [('Train', train), ('Test', test), ('Validation', validation)]:
    # Duplicates by NCBIGeneID
    id_dups = df['NCBIGeneID'].duplicated().sum()
    # Duplicates by sequence
    seq_dups = df['NucleotideSequence'].duplicated().sum()
    # Full duplicates
    full_dups = df.duplicated().sum()
    
    print(f"\n{name}:")
    print(f"  Duplicate NCBIGeneID: {id_dups}")
    print(f"  Duplicate sequences: {seq_dups}")
    print(f"  Fully duplicate rows: {full_dups}")

# ============================================
# Issue 5: Data Leakage
# ============================================
print("\n" + "=" * 60)
print("Issue 5: Data Leakage Check")
print("=" * 60)

# Check overlap between sets
train_ids = set(train['NCBIGeneID'])
test_ids = set(test['NCBIGeneID'])
val_ids = set(validation['NCBIGeneID'])

train_test_overlap = train_ids & test_ids
train_val_overlap = train_ids & val_ids
test_val_overlap = test_ids & val_ids

print(f"\nNCBIGeneID overlap:")
print(f"  Train & Test: {len(train_test_overlap)} genes")
print(f"  Train & Validation: {len(train_val_overlap)} genes")
print(f"  Test & Validation: {len(test_val_overlap)} genes")

# Sequence overlap
train_seqs = set(train['NucleotideSequence'])
test_seqs = set(test['NucleotideSequence'])
val_seqs = set(validation['NucleotideSequence'])

seq_train_test = train_seqs & test_seqs
seq_train_val = train_seqs & val_seqs

print(f"\nNucleotide sequence overlap:")
print(f"  Train & Test: {len(seq_train_test)} sequences")
print(f"  Train & Validation: {len(seq_train_val)} sequences")

# ============================================
# Issue 6: Constant Feature
# ============================================
print("\n" + "=" * 60)
print("Issue 6: Constant Features")
print("=" * 60)

all_data = pd.concat([train, test, validation])
for col in all_data.columns:
    n_unique = all_data[col].nunique()
    if n_unique == 1:
        print(f"  {col}: CONSTANT value = '{all_data[col].iloc[0]}'")

# ============================================
# Issue 7: Variable Sequence Lengths
# ============================================
print("\n" + "=" * 60)
print("Issue 7: Variable Sequence Lengths")
print("=" * 60)

def get_seq_length(seq):
    if pd.isna(seq):
        return 0
    return len(str(seq).strip('<>'))

train['seq_length'] = train['NucleotideSequence'].apply(get_seq_length)
print(f"\nSequence length statistics (Train):")
print(f"  Minimum: {train['seq_length'].min()}")
print(f"  Maximum: {train['seq_length'].max()}")
print(f"  Mean: {train['seq_length'].mean():.0f}")
print(f"  Median: {train['seq_length'].median():.0f}")
print(f"  Std Dev: {train['seq_length'].std():.0f}")

# Distribution by length categories
bins = [0, 50, 100, 200, 500, 1000, float('inf')]
labels = ['0-50', '51-100', '101-200', '201-500', '501-1000', '1000+']
train['length_bin'] = pd.cut(train['seq_length'], bins=bins, labels=labels)
print(f"\nSequence length distribution:")
print(train['length_bin'].value_counts().sort_index())

# ============================================
# Issue 8: Invalid Characters in Sequence
# ============================================
print("\n" + "=" * 60)
print("Issue 8: Invalid Characters in DNA/RNA Sequence")
print("=" * 60)

valid_nucleotides = set('ACGTU')

def check_invalid_chars(seq):
    if pd.isna(seq):
        return set()
    seq_clean = str(seq).strip('<>')
    return set(seq_clean) - valid_nucleotides

invalid_chars = set()
for seq in train['NucleotideSequence']:
    invalid_chars.update(check_invalid_chars(seq))

if invalid_chars:
    print(f"Found invalid characters: {invalid_chars}")
else:
    print("All sequences contain only ACGTU [OK]")

# ============================================
# Issue 9: Outliers in Sequence Length
# ============================================
print("\n" + "=" * 60)
print("Issue 9: Outliers in Sequence Length")
print("=" * 60)

Q1 = train['seq_length'].quantile(0.25)
Q3 = train['seq_length'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = train[(train['seq_length'] < lower_bound) | (train['seq_length'] > upper_bound)]
print(f"Lower bound: {lower_bound:.0f}")
print(f"Upper bound: {upper_bound:.0f}")
print(f"Number of outliers: {len(outliers)} ({len(outliers)/len(train)*100:.1f}%)")

# ============================================
# Issue 10: Sequence Length by Class
# ============================================
print("\n" + "=" * 60)
print("Issue 10: Sequence Length by Class")
print("=" * 60)

print("\nMean sequence length per GeneType:")
for gt in train['GeneType'].unique():
    mean_len = train[train['GeneType'] == gt]['seq_length'].mean()
    print(f"  {gt}: {mean_len:.0f}")

# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("SUMMARY OF ISSUES AND RECOMMENDATIONS")
print("=" * 60)

print("""
[CRITICAL ISSUES]
1. Extreme class imbalance (ratio ~3406:1)
2. scRNA class MISSING completely from Validation (0 samples!)
3. Very rare classes: scRNA (3), snRNA, rRNA

[MODERATE ISSUES]
4. GeneGroupMethod is CONSTANT - can be removed
5. High variance in sequence lengths (28-896 bases)
6. ~17% of sequences are outliers in length

[POSITIVE FINDINGS]
+ No missing values
+ No duplicates detected
+ No data leakage between sets
+ All sequence characters are valid (ACGTU)

[RECOMMENDATIONS]
1. For class imbalance:
   - Use SMOTE or other oversampling for rare classes
   - Use class weights in loss function
   - Consider undersampling majority classes
   - Use stratified sampling for cross-validation

2. For missing scRNA in validation:
   - Move 1-2 samples from train to validation
   - Or remove scRNA class entirely (only 4 total)
   - Or merge similar RNA classes

3. For variable sequence lengths:
   - Use padding/truncation to fixed length
   - Use sequence embedding models (like DNA-BERT)
   - Consider using length as additional feature

4. For rare classes:
   - Consider hierarchical classification
   - Group small RNA types (snoRNA, snRNA, scRNA) together
   - Use focal loss or other imbalance-aware losses
""")
