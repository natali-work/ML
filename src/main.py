"""
Main Pipeline Script
Orchestrates the complete gene classification ML pipeline.
"""

import sys
from pathlib import Path

# Add src to path for imports
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

from data_loader import load_and_clean_data
from preprocessing import preprocess_data
from features import prepare_ml_data
from model import train_and_evaluate, compare_models, XGBOOST_AVAILABLE


def run_pipeline(data_dir: str = None, 
                 models_dir: str = None,
                 train_rf: bool = True,
                 train_xgb: bool = True) -> dict:
    """
    Run the complete ML pipeline.
    
    Args:
        data_dir: Path to directory containing CSV files
        models_dir: Path to directory for saving models
        train_rf: Whether to train Random Forest
        train_xgb: Whether to train XGBoost
        
    Returns:
        Dictionary with results
    """
    # Set default paths
    if data_dir is None:
        data_dir = src_dir.parent
    else:
        data_dir = Path(data_dir)
    
    if models_dir is None:
        models_dir = data_dir / "models"
    else:
        models_dir = Path(models_dir)
    
    # Ensure models directory exists
    models_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("GENE CLASSIFICATION ML PIPELINE")
    print("=" * 60)
    print(f"\nData directory: {data_dir}")
    print(f"Models directory: {models_dir}")
    
    # =========================================
    # STEP 1: Load and Clean Data
    # =========================================
    print("\n" + "=" * 60)
    print("STEP 1: LOADING AND CLEANING DATA")
    print("=" * 60)
    
    df = load_and_clean_data(data_dir)
    
    # =========================================
    # STEP 2: Preprocess Data
    # =========================================
    print("\n" + "=" * 60)
    print("STEP 2: PREPROCESSING (Fix Leakage, Merge Classes)")
    print("=" * 60)
    
    train_df, test_df, val_df = preprocess_data(
        df,
        merge_classes=True,  # Merge scRNA, snRNA, rRNA -> small_RNA
        test_size=0.15,
        val_size=0.15,
        random_state=42
    )
    
    # =========================================
    # STEP 3: Feature Engineering
    # =========================================
    print("\n" + "=" * 60)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 60)
    
    X_train, X_test, X_val, y_train, y_test, y_val, feature_names, class_names = \
        prepare_ml_data(train_df, test_df, val_df)
    
    # Store results
    results = {
        'data_shapes': {
            'X_train': X_train.shape,
            'X_test': X_test.shape,
            'X_val': X_val.shape
        },
        'n_features': len(feature_names),
        'n_classes': len(class_names),
        'class_names': class_names,
        'models': {}
    }
    
    # =========================================
    # STEP 4: Train Models
    # =========================================
    print("\n" + "=" * 60)
    print("STEP 4: TRAINING MODELS")
    print("=" * 60)
    
    # Train Random Forest
    if train_rf:
        print("\n--- Training Random Forest ---")
        rf_classifier, rf_results = train_and_evaluate(
            X_train, y_train, 
            X_test, y_test, 
            X_val, y_val,
            feature_names, class_names,
            model_type='random_forest',
            save_dir=models_dir
        )
        results['models']['random_forest'] = rf_results
    
    # Train XGBoost
    if train_xgb and XGBOOST_AVAILABLE:
        print("\n--- Training XGBoost ---")
        xgb_classifier, xgb_results = train_and_evaluate(
            X_train, y_train, 
            X_test, y_test, 
            X_val, y_val,
            feature_names, class_names,
            model_type='xgboost',
            save_dir=models_dir
        )
        results['models']['xgboost'] = xgb_results
    elif train_xgb and not XGBOOST_AVAILABLE:
        print("\nSkipping XGBoost (not installed)")
    
    # =========================================
    # STEP 5: Compare Models
    # =========================================
    if train_rf and train_xgb and XGBOOST_AVAILABLE:
        print("\n" + "=" * 60)
        print("STEP 5: MODEL COMPARISON")
        print("=" * 60)
        compare_models(rf_results, xgb_results)
    
    # =========================================
    # SUMMARY
    # =========================================
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    
    print(f"\nSaved files in {models_dir}:")
    for f in models_dir.glob("*"):
        print(f"  - {f.name}")
    
    # Print best results
    print("\nBest Results Summary:")
    for model_name, model_results in results['models'].items():
        test_f1 = model_results['test']['macro_f1']
        val_f1 = model_results['validation']['macro_f1']
        print(f"  {model_name}:")
        print(f"    - Validation Macro F1: {val_f1:.4f}")
        print(f"    - Test Macro F1: {test_f1:.4f}")
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Gene Classification ML Pipeline')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Path to data directory (default: parent of src)')
    parser.add_argument('--models-dir', type=str, default=None,
                       help='Path to save models (default: data_dir/models)')
    parser.add_argument('--no-rf', action='store_true',
                       help='Skip Random Forest training')
    parser.add_argument('--no-xgb', action='store_true',
                       help='Skip XGBoost training')
    
    args = parser.parse_args()
    
    results = run_pipeline(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        train_rf=not args.no_rf,
        train_xgb=not args.no_xgb
    )
    
    return results


if __name__ == "__main__":
    main()
