# -*- coding: utf-8 -*-
"""
Gene Classification Dataset - Visualization Script
Generates comprehensive visualizations for understanding the genetic data
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Hebrew font support (if available)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 150

print("Loading data...")

# Load datasets
train_df = pd.read_csv('../train.csv')
test_df = pd.read_csv('../test.csv')
val_df = pd.read_csv('../validation.csv')

print(f"Train: {len(train_df)} samples")
print(f"Test: {len(test_df)} samples")
print(f"Validation: {len(val_df)} samples")

# Add sequence length column
train_df['SeqLength'] = train_df['NucleotideSequence'].apply(lambda x: len(str(x).replace('<', '').replace('>', '')))
test_df['SeqLength'] = test_df['NucleotideSequence'].apply(lambda x: len(str(x).replace('<', '').replace('>', '')))
val_df['SeqLength'] = val_df['NucleotideSequence'].apply(lambda x: len(str(x).replace('<', '').replace('>', '')))

# Color palette for gene types
gene_colors = {
    'PSEUDO': '#E74C3C',
    'BIOLOGICAL_REGION': '#3498DB',
    'ncRNA': '#2ECC71',
    'snoRNA': '#9B59B6',
    'PROTEIN_CODING': '#F39C12',
    'tRNA': '#1ABC9C',
    'OTHER': '#7F8C8D',
    'rRNA': '#E91E63',
    'snRNA': '#00BCD4',
    'scRNA': '#FF5722'
}

# ============================================
# 1. Gene Type Distribution - Bar Chart
# ============================================
print("\n1. Creating Gene Type Distribution chart...")

fig, ax = plt.subplots(figsize=(14, 8))
gene_counts = train_df['GeneType'].value_counts()
colors = [gene_colors.get(g, '#333333') for g in gene_counts.index]

bars = ax.bar(gene_counts.index, gene_counts.values, color=colors, edgecolor='white', linewidth=1.5)

# Add value labels on bars
for bar, count in zip(bars, gene_counts.values):
    height = bar.get_height()
    ax.annotate(f'{count:,}\n({count/len(train_df)*100:.1f}%)',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5), textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Gene Type', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Gene Types in Training Set', fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
ax.set_ylim(0, max(gene_counts.values) * 1.15)

plt.tight_layout()
plt.savefig('01_gene_type_distribution.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   Saved: 01_gene_type_distribution.png")

# ============================================
# 2. Class Imbalance Pie Chart
# ============================================
print("\n2. Creating Class Imbalance Pie Chart...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Pie chart - all classes
ax1 = axes[0]
colors_pie = [gene_colors.get(g, '#333333') for g in gene_counts.index]
wedges, texts, autotexts = ax1.pie(gene_counts.values, labels=gene_counts.index, 
                                    autopct='%1.1f%%', colors=colors_pie,
                                    explode=[0.05 if i < 2 else 0 for i in range(len(gene_counts))],
                                    shadow=True, startangle=90)
ax1.set_title('Gene Type Distribution\n(All Classes)', fontsize=14, fontweight='bold')

# Grouped pie - showing imbalance
ax2 = axes[1]
dominant = gene_counts.iloc[:2].sum()
minority = gene_counts.iloc[2:].sum()
labels_grouped = ['PSEUDO + BIOLOGICAL_REGION\n(Dominant Classes)', 'All Other Classes\n(Minority Classes)']
sizes_grouped = [dominant, minority]
colors_grouped = ['#E74C3C', '#3498DB']
explode_grouped = (0.05, 0)

wedges2, texts2, autotexts2 = ax2.pie(sizes_grouped, labels=labels_grouped, autopct='%1.1f%%',
                                       colors=colors_grouped, explode=explode_grouped,
                                       shadow=True, startangle=90)
ax2.set_title('Class Imbalance Overview\n(Dominant vs Minority)', fontsize=14, fontweight='bold')

for autotext in autotexts + autotexts2:
    autotext.set_fontsize(11)
    autotext.set_fontweight('bold')

plt.tight_layout()
plt.savefig('02_class_imbalance_pie.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   Saved: 02_class_imbalance_pie.png")

# ============================================
# 3. Train/Test/Validation Comparison
# ============================================
print("\n3. Creating Dataset Split Comparison...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: Sample counts
ax1 = axes[0]
datasets = ['Train', 'Test', 'Validation']
counts = [len(train_df), len(test_df), len(val_df)]
colors_ds = ['#3498DB', '#2ECC71', '#F39C12']

bars = ax1.bar(datasets, counts, color=colors_ds, edgecolor='white', linewidth=2)
for bar, count, total in zip(bars, counts, counts):
    ax1.annotate(f'{count:,}\n({count/sum(counts)*100:.1f}%)',
                 xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                 xytext=(0, 5), textcoords="offset points",
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
ax1.set_title('Dataset Split Distribution', fontsize=14, fontweight='bold')
ax1.set_ylim(0, max(counts) * 1.2)

# Right: Grouped bar comparison
ax2 = axes[1]
gene_types = gene_counts.index.tolist()
train_pcts = [(train_df['GeneType'] == g).mean() * 100 for g in gene_types]
test_pcts = [(test_df['GeneType'] == g).mean() * 100 for g in gene_types]
val_pcts = [(val_df['GeneType'] == g).mean() * 100 for g in gene_types]

x = np.arange(len(gene_types))
width = 0.25

bars1 = ax2.bar(x - width, train_pcts, width, label='Train', color='#3498DB', edgecolor='white')
bars2 = ax2.bar(x, test_pcts, width, label='Test', color='#2ECC71', edgecolor='white')
bars3 = ax2.bar(x + width, val_pcts, width, label='Validation', color='#F39C12', edgecolor='white')

ax2.set_xlabel('Gene Type', fontsize=12, fontweight='bold')
ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax2.set_title('Gene Type Distribution Across Datasets', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(gene_types, rotation=45, ha='right', fontsize=9)
ax2.legend(loc='upper right', fontsize=10)
ax2.set_ylim(0, max(train_pcts) * 1.1)

plt.tight_layout()
plt.savefig('03_dataset_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   Saved: 03_dataset_comparison.png")

# ============================================
# 4. Sequence Length Distribution
# ============================================
print("\n4. Creating Sequence Length Distribution...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Overall distribution
ax1 = axes[0, 0]
ax1.hist(train_df['SeqLength'], bins=100, color='#3498DB', alpha=0.7, edgecolor='white')
ax1.axvline(train_df['SeqLength'].median(), color='red', linestyle='--', linewidth=2, label=f'Median: {train_df["SeqLength"].median():.0f}')
ax1.axvline(train_df['SeqLength'].mean(), color='orange', linestyle='--', linewidth=2, label=f'Mean: {train_df["SeqLength"].mean():.0f}')
ax1.set_xlabel('Sequence Length (nucleotides)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax1.set_title('Overall Sequence Length Distribution', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)

# Log scale distribution
ax2 = axes[0, 1]
ax2.hist(train_df['SeqLength'], bins=100, color='#2ECC71', alpha=0.7, edgecolor='white')
ax2.set_yscale('log')
ax2.set_xlabel('Sequence Length (nucleotides)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Frequency (log scale)', fontsize=11, fontweight='bold')
ax2.set_title('Sequence Length Distribution (Log Scale)', fontsize=13, fontweight='bold')

# Box plot by gene type
ax3 = axes[1, 0]
gene_type_order = gene_counts.index.tolist()
box_colors = [gene_colors.get(g, '#333333') for g in gene_type_order]
bp = ax3.boxplot([train_df[train_df['GeneType'] == g]['SeqLength'] for g in gene_type_order],
                  labels=gene_type_order, patch_artist=True, showfliers=False)
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax3.set_xlabel('Gene Type', fontsize=11, fontweight='bold')
ax3.set_ylabel('Sequence Length', fontsize=11, fontweight='bold')
ax3.set_title('Sequence Length by Gene Type (Outliers Hidden)', fontsize=13, fontweight='bold')
plt.setp(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=9)

# Statistics table
ax4 = axes[1, 1]
ax4.axis('off')
stats_data = []
for g in gene_type_order:
    subset = train_df[train_df['GeneType'] == g]['SeqLength']
    stats_data.append([g, f'{subset.min():,}', f'{subset.median():.0f}', f'{subset.mean():.0f}', f'{subset.max():,}'])

table = ax4.table(cellText=stats_data,
                   colLabels=['Gene Type', 'Min', 'Median', 'Mean', 'Max'],
                   loc='center',
                   cellLoc='center',
                   colColours=['#E8E8E8']*5)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)
ax4.set_title('Sequence Length Statistics by Gene Type', fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('04_sequence_length_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   Saved: 04_sequence_length_analysis.png")

# ============================================
# 5. Heatmap - Distribution Consistency
# ============================================
print("\n5. Creating Distribution Consistency Heatmap...")

fig, ax = plt.subplots(figsize=(12, 8))

# Create percentage matrix
gene_types_all = list(gene_colors.keys())
data_matrix = []
for g in gene_types_all:
    train_pct = (train_df['GeneType'] == g).mean() * 100
    test_pct = (test_df['GeneType'] == g).mean() * 100
    val_pct = (val_df['GeneType'] == g).mean() * 100
    data_matrix.append([train_pct, test_pct, val_pct])

df_heatmap = pd.DataFrame(data_matrix, index=gene_types_all, columns=['Train', 'Test', 'Validation'])

sns.heatmap(df_heatmap, annot=True, fmt='.2f', cmap='YlOrRd', 
            linewidths=0.5, ax=ax, cbar_kws={'label': 'Percentage (%)'})
ax.set_title('Gene Type Distribution (%) Across All Datasets', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
ax.set_ylabel('Gene Type', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('05_distribution_heatmap.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   Saved: 05_distribution_heatmap.png")

# ============================================
# 6. Nucleotide Composition Analysis
# ============================================
print("\n6. Creating Nucleotide Composition Analysis...")

def calculate_nucleotide_composition(sequence):
    """Calculate percentage of each nucleotide"""
    seq = str(sequence).upper().replace('<', '').replace('>', '')
    total = len(seq)
    if total == 0:
        return {'A': 0, 'T': 0, 'G': 0, 'C': 0}
    return {
        'A': seq.count('A') / total * 100,
        'T': seq.count('T') / total * 100,
        'G': seq.count('G') / total * 100,
        'C': seq.count('C') / total * 100
    }

# Sample for faster processing
sample_df = train_df.sample(min(5000, len(train_df)), random_state=42)
compositions = sample_df['NucleotideSequence'].apply(calculate_nucleotide_composition)
compositions_df = pd.DataFrame(compositions.tolist())
sample_df = pd.concat([sample_df.reset_index(drop=True), compositions_df], axis=1)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Overall nucleotide distribution
ax1 = axes[0, 0]
nucleotides = ['A', 'T', 'G', 'C']
overall_means = [sample_df[n].mean() for n in nucleotides]
colors_nuc = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
bars = ax1.bar(nucleotides, overall_means, color=colors_nuc, edgecolor='white', linewidth=2)
for bar, val in zip(bars, overall_means):
    ax1.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 5), textcoords="offset points", ha='center', fontsize=12, fontweight='bold')
ax1.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax1.set_title('Average Nucleotide Composition', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 40)

# GC Content distribution
ax2 = axes[0, 1]
sample_df['GC_Content'] = sample_df['G'] + sample_df['C']
ax2.hist(sample_df['GC_Content'], bins=50, color='#9B59B6', alpha=0.7, edgecolor='white')
ax2.axvline(sample_df['GC_Content'].mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {sample_df["GC_Content"].mean():.1f}%')
ax2.set_xlabel('GC Content (%)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax2.set_title('GC Content Distribution', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)

# GC Content by Gene Type
ax3 = axes[1, 0]
gc_by_type = sample_df.groupby('GeneType')['GC_Content'].mean().sort_values(ascending=False)
colors_gc = [gene_colors.get(g, '#333333') for g in gc_by_type.index]
bars = ax3.bar(gc_by_type.index, gc_by_type.values, color=colors_gc, edgecolor='white')
ax3.axhline(50, color='red', linestyle='--', alpha=0.5, label='50% (balanced)')
for bar, val in zip(bars, gc_by_type.values):
    ax3.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
ax3.set_xlabel('Gene Type', fontsize=12, fontweight='bold')
ax3.set_ylabel('Average GC Content (%)', fontsize=12, fontweight='bold')
ax3.set_title('Average GC Content by Gene Type', fontsize=14, fontweight='bold')
plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
ax3.legend()

# Stacked bar - composition by gene type
ax4 = axes[1, 1]
comp_by_type = sample_df.groupby('GeneType')[nucleotides].mean()
comp_by_type = comp_by_type.loc[gene_counts.index[:8]]  # Top 8 types

bottom = np.zeros(len(comp_by_type))
for i, nuc in enumerate(nucleotides):
    ax4.bar(comp_by_type.index, comp_by_type[nuc], bottom=bottom, 
            label=nuc, color=colors_nuc[i], edgecolor='white')
    bottom += comp_by_type[nuc].values

ax4.set_xlabel('Gene Type', fontsize=12, fontweight='bold')
ax4.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax4.set_title('Nucleotide Composition by Gene Type', fontsize=14, fontweight='bold')
ax4.legend(title='Nucleotide', loc='upper right')
plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('06_nucleotide_composition.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   Saved: 06_nucleotide_composition.png")

# ============================================
# 7. Rare Classes Analysis
# ============================================
print("\n7. Creating Rare Classes Analysis...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Rare classes detail
ax1 = axes[0]
rare_types = ['tRNA', 'OTHER', 'rRNA', 'snRNA', 'scRNA']
train_counts = [sum(train_df['GeneType'] == t) for t in rare_types]
test_counts = [sum(test_df['GeneType'] == t) for t in rare_types]
val_counts = [sum(val_df['GeneType'] == t) for t in rare_types]

x = np.arange(len(rare_types))
width = 0.25

ax1.bar(x - width, train_counts, width, label='Train', color='#3498DB')
ax1.bar(x, test_counts, width, label='Test', color='#2ECC71')
ax1.bar(x + width, val_counts, width, label='Validation', color='#F39C12')

ax1.set_xlabel('Gene Type (Rare Classes)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
ax1.set_title('Rare Classes Distribution Across Datasets', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(rare_types)
ax1.legend()

# Annotations
for i, (tr, te, va) in enumerate(zip(train_counts, test_counts, val_counts)):
    ax1.annotate(str(tr), xy=(i - width, tr), xytext=(0, 2), textcoords="offset points", ha='center', fontsize=8)
    ax1.annotate(str(te), xy=(i, te), xytext=(0, 2), textcoords="offset points", ha='center', fontsize=8)
    ax1.annotate(str(va), xy=(i + width, va), xytext=(0, 2), textcoords="offset points", ha='center', fontsize=8)

# scRNA problem highlight
ax2 = axes[1]
scRNA_data = {'Train': 3, 'Test': 1, 'Validation': 0}
bars = ax2.bar(scRNA_data.keys(), scRNA_data.values(), color=['#E74C3C', '#F39C12', '#7F8C8D'], edgecolor='white')
for bar, val in zip(bars, scRNA_data.values()):
    ax2.annotate(f'{val}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 5), textcoords="offset points", ha='center', fontsize=14, fontweight='bold')
ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
ax2.set_title('scRNA Class: Critical Data Issue\n(Only 4 samples total!)', fontsize=14, fontweight='bold', color='#E74C3C')
ax2.set_ylim(0, 5)

# Add warning box
ax2.text(0.5, 0.7, '⚠️ WARNING\nscRNA has only 4 samples total\n0 samples in validation set!\nConsider removing or merging this class',
         transform=ax2.transAxes, fontsize=11, verticalalignment='center', horizontalalignment='center',
         bbox=dict(boxstyle='round', facecolor='#FFEBEE', edgecolor='#E74C3C', linewidth=2))

plt.tight_layout()
plt.savefig('07_rare_classes_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   Saved: 07_rare_classes_analysis.png")

# ============================================
# 8. Summary Dashboard
# ============================================
print("\n8. Creating Summary Dashboard...")

fig = plt.figure(figsize=(20, 14))

# Title
fig.suptitle('Gene Classification Dataset - Summary Dashboard', fontsize=20, fontweight='bold', y=0.98)

# 1. Dataset Overview
ax1 = fig.add_subplot(2, 3, 1)
datasets = ['Train', 'Test', 'Validation', 'Total']
counts = [len(train_df), len(test_df), len(val_df), len(train_df)+len(test_df)+len(val_df)]
colors_ds = ['#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
bars = ax1.bar(datasets, counts, color=colors_ds, edgecolor='white')
for bar, count in zip(bars, counts):
    ax1.annotate(f'{count:,}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')
ax1.set_ylabel('Samples', fontweight='bold')
ax1.set_title('Dataset Sizes', fontsize=12, fontweight='bold')

# 2. Top 5 Gene Types
ax2 = fig.add_subplot(2, 3, 2)
top5 = gene_counts.head(5)
colors_t5 = [gene_colors.get(g, '#333333') for g in top5.index]
wedges, texts, autotexts = ax2.pie(top5.values, labels=top5.index, autopct='%1.1f%%', 
                                    colors=colors_t5, startangle=90)
ax2.set_title('Top 5 Gene Types (Training)', fontsize=12, fontweight='bold')

# 3. Class Imbalance Ratio
ax3 = fig.add_subplot(2, 3, 3)
max_class = gene_counts.max()
ratios = max_class / gene_counts
ax3.barh(gene_counts.index, ratios, color=[gene_colors.get(g, '#333333') for g in gene_counts.index])
ax3.set_xlabel('Imbalance Ratio (relative to PSEUDO)', fontweight='bold')
ax3.set_title('Class Imbalance Ratios', fontsize=12, fontweight='bold')
ax3.invert_yaxis()

# 4. Sequence Length Summary
ax4 = fig.add_subplot(2, 3, 4)
seq_stats = ['Min', 'Median', 'Mean', 'Max']
seq_vals = [train_df['SeqLength'].min(), train_df['SeqLength'].median(), 
            train_df['SeqLength'].mean(), train_df['SeqLength'].max()]
bars = ax4.bar(seq_stats, seq_vals, color=['#1ABC9C', '#3498DB', '#9B59B6', '#E74C3C'])
for bar, val in zip(bars, seq_vals):
    ax4.annotate(f'{val:,.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')
ax4.set_ylabel('Nucleotides', fontweight='bold')
ax4.set_title('Sequence Length Statistics', fontsize=12, fontweight='bold')
ax4.set_yscale('log')

# 5. Key Insights Box
ax5 = fig.add_subplot(2, 3, 5)
ax5.axis('off')
insights = """
KEY INSIGHTS

✓ Total Samples: 35,496
✓ Gene Types: 10 classes
✓ Dominant Classes: PSEUDO + BIOLOGICAL_REGION = 76%

⚠️ CHALLENGES:
• Severe class imbalance (3,407:1 ratio)
• scRNA has only 4 samples (unusable)
• 5 classes have < 3% representation

💡 RECOMMENDATIONS:
• Use stratified sampling
• Consider class weights or SMOTE
• Remove or merge scRNA class
• Focus on top 5-6 classes initially
"""
ax5.text(0.1, 0.95, insights, transform=ax5.transAxes, fontsize=11, 
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#F8F9FA', edgecolor='#DEE2E6', linewidth=2))
ax5.set_title('Summary & Recommendations', fontsize=12, fontweight='bold')

# 6. GC Content by Class
ax6 = fig.add_subplot(2, 3, 6)
gc_means = sample_df.groupby('GeneType')['GC_Content'].mean().sort_values()
colors_gc = [gene_colors.get(g, '#333333') for g in gc_means.index]
ax6.barh(gc_means.index, gc_means.values, color=colors_gc)
ax6.axvline(50, color='red', linestyle='--', alpha=0.5, label='50% baseline')
ax6.set_xlabel('GC Content (%)', fontweight='bold')
ax6.set_title('Average GC Content by Gene Type', fontsize=12, fontweight='bold')
ax6.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('08_summary_dashboard.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   Saved: 08_summary_dashboard.png")

print("\n" + "="*60)
print("All visualizations generated successfully!")
print("="*60)
