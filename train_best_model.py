"""
Train and save the best fraud detection model
"""
import argparse
from fraud_model import FraudDetectionModel

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train and compare fraud detection models')
parser.add_argument('--data', type=str, default='/data/creditcard.csv',
                   help='Path to credit card data CSV file')
parser.add_argument('--output', type=str, default='/data/models/best_fraud_detector.pkl',
                   help='Path to save the best model')
args = parser.parse_args()

# Model configurations to try
configs = [
    {'model_type': 'logistic', 'sampling_strategy': 'none', 'name': 'Logistic (No Sampling)'},
    {'model_type': 'weighted_logistic', 'sampling_strategy': 'none', 'name': 'Logistic (Weighted)'},
    {'model_type': 'logistic', 'sampling_strategy': 'smote', 'name': 'Logistic + SMOTE'},
    {'model_type': 'logistic', 'sampling_strategy': 'undersample', 'name': 'Logistic + Undersample'},
    {'model_type': 'random_forest', 'sampling_strategy': 'smote', 'name': 'Random Forest + SMOTE'},
]

print("Training and comparing models...")
print("="*80)

best_model = None
best_f1 = 0
best_config = None
all_results = []

for config in configs:
    print(f"\nTraining: {config['name']}")
    print("-"*80)
    
    # Initialize model
    model = FraudDetectionModel(
        model_type=config['model_type'],
        sampling_strategy=config['sampling_strategy']
    )
    
    # Train
    model.load_data(args.data)
    model.preprocess()
    model.build_model()
    model.train()
    
    # Evaluate
    metrics = model.evaluate(verbose=False)
    metrics['name'] = config['name']
    all_results.append(metrics)
    
    print(f"F1: {metrics['f1']:.4f} | Recall: {metrics['recall']:.4f} | Precision: {metrics['precision']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Track best model
    if metrics['f1'] > best_f1:
        best_f1 = metrics['f1']
        best_model = model
        best_config = config

# Print comparison
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)
print(f"{'Model':<30} {'F1':>8} {'Recall':>8} {'Precision':>10} {'ROC-AUC':>10}")
print("-"*80)

for result in all_results:
    print(f"{result['name']:<30} {result['f1']:>8.4f} {result['recall']:>8.4f} "
          f"{result['precision']:>10.4f} {result['roc_auc']:>10.4f}")

# Save best model
print("\n" + "="*80)
print(f"BEST MODEL: {best_config['name']}")
print(f"F1 Score: {best_f1:.4f}")
print("="*80)

best_model.save_model(args.output)

# Also evaluate best model with full details
print("\nDetailed evaluation of best model:")
best_model.evaluate(verbose=True)

print(f"\n✓ Best model saved to {args.output}")
