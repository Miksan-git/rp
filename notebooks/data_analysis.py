"""
Data Analysis Script

Analyzes the dataset to understand:
- Feature distributions
- Class distributions
- Data quality
- Missing values
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def analyze_dataset(dataset_path: str, output_dir: str = "results/analysis"):
    """
    Comprehensive dataset analysis.
    
    Args:
        dataset_path: Path to CSV dataset
        output_dir: Directory to save analysis results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading dataset...")
    df = pd.read_csv(dataset_path)
    print(f"Dataset shape: {df.shape}")
    
    # Basic statistics
    print("\n" + "="*80)
    print("DATASET OVERVIEW")
    print("="*80)
    print(f"Total samples: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    
    # Missing values
    print("\n" + "="*80)
    print("MISSING VALUES")
    print("="*80)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing Percentage': missing_pct
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    if len(missing_df) > 0:
        print(missing_df)
    else:
        print("No missing values found!")
    
    # Feature distributions
    print("\n" + "="*80)
    print("NUMERICAL FEATURES")
    print("="*80)
    numerical_cols = ['Age', 'Weight']
    for col in numerical_cols:
        if col in df.columns:
            print(f"\n{col}:")
            print(df[col].describe())
    
    # Categorical feature distributions
    print("\n" + "="*80)
    print("CATEGORICAL FEATURES")
    print("="*80)
    categorical_cols = [
        'Breed', 'Medical History', 'Genetic Predispositions',
        'Current Medications', 'Diet', 'Lifestyle', 'Environment',
        'Vaccination Status', 'Neutering Status', 'Living Conditions',
        'Disease', 'Stage'
    ]
    
    for col in categorical_cols:
        if col in df.columns:
            print(f"\n{col}:")
            value_counts = df[col].value_counts()
            print(f"  Unique values: {df[col].nunique()}")
            print(f"  Top 5 values:")
            for val, count in value_counts.head(5).items():
                print(f"    {val}: {count} ({count/len(df)*100:.2f}%)")
    
    # Target distributions
    print("\n" + "="*80)
    print("TARGET DISTRIBUTIONS")
    print("="*80)
    
    # Conventional treatment
    if 'Conventional Treatment' in df.columns:
        print("\nConventional Treatment:")
        conv_counts = df['Conventional Treatment'].value_counts()
        print(f"  Unique treatments: {df['Conventional Treatment'].nunique()}")
        for treatment, count in conv_counts.items():
            print(f"    {treatment}: {count} ({count/len(df)*100:.2f}%)")
    
    # Natural remedies (multi-label)
    if 'Natural Remedies' in df.columns:
        print("\nNatural Remedies:")
        # Parse comma-separated remedies
        all_remedies = []
        for remedies_str in df['Natural Remedies'].dropna():
            remedies = [r.strip() for r in str(remedies_str).split(',')]
            all_remedies.extend(remedies)
        
        remedy_counts = pd.Series(all_remedies).value_counts()
        print(f"  Unique remedies: {len(remedy_counts)}")
        print(f"  Top 10 remedies:")
        for remedy, count in remedy_counts.head(10).items():
            print(f"    {remedy}: {count}")
    
    # Visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Age and Weight distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if 'Age' in df.columns:
        axes[0].hist(df['Age'], bins=20, edgecolor='black')
        axes[0].set_title('Age Distribution')
        axes[0].set_xlabel('Age (years)')
        axes[0].set_ylabel('Frequency')
    
    if 'Weight' in df.columns:
        axes[1].hist(df['Weight'], bins=20, edgecolor='black')
        axes[1].set_title('Weight Distribution')
        axes[1].set_xlabel('Weight (kg)')
        axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'numerical_distributions.png'), dpi=300)
    plt.close()
    
    # Disease distribution
    if 'Disease' in df.columns:
        plt.figure(figsize=(10, 6))
        disease_counts = df['Disease'].value_counts()
        plt.barh(range(len(disease_counts)), disease_counts.values)
        plt.yticks(range(len(disease_counts)), disease_counts.index)
        plt.xlabel('Frequency')
        plt.title('Disease Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'disease_distribution.png'), dpi=300)
        plt.close()
    
    # Conventional treatment distribution
    if 'Conventional Treatment' in df.columns:
        plt.figure(figsize=(10, 6))
        treatment_counts = df['Conventional Treatment'].value_counts()
        plt.barh(range(len(treatment_counts)), treatment_counts.values)
        plt.yticks(range(len(treatment_counts)), treatment_counts.index)
        plt.xlabel('Frequency')
        plt.title('Conventional Treatment Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'conventional_treatment_distribution.png'), dpi=300)
        plt.close()
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    dataset_path = "../Refined_Book_Aligned_Dog_Treatment_Dataset.csv"
    analyze_dataset(dataset_path)

