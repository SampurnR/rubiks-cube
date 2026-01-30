"""
Make fraud predictions using the trained model
"""
import pandas as pd
import numpy as np
from fraud_model import FraudDetectionModel
import argparse


def predict_from_csv(model_path, data_path, output_path='predictions.csv'):
    """
    Make predictions on a CSV file of transactions.
    
    Args:
        model_path: Path to saved model
        data_path: Path to CSV with transactions
        output_path: Path to save predictions
    """
    print("Loading model...")
    model = FraudDetectionModel()
    model.load_model(model_path)
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Check if it has Class column (labeled data)
    has_labels = 'Class' in df.columns
    
    if has_labels:
        X = df.drop('Class', axis=1)
        y_true = df['Class']
    else:
        X = df
    
    print(f"Making predictions on {len(X)} transactions...")
    
    # Get predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Create results dataframe
    results = pd.DataFrame({
        'prediction': predictions,
        'fraud_probability': probabilities,
        'is_fraud': predictions == 1
    })
    
    if has_labels:
        results['actual'] = y_true
        results['correct'] = results['prediction'] == results['actual']
        
        accuracy = results['correct'].mean()
        fraud_detected = (results['prediction'] == 1).sum()
        actual_fraud = (results['actual'] == 1).sum()
        
        print(f"\nResults:")
        print(f"  Total transactions: {len(results)}")
        print(f"  Predicted fraud: {fraud_detected}")
        print(f"  Actual fraud: {actual_fraud}")
        print(f"  Accuracy: {accuracy:.4f}")
    else:
        fraud_detected = (results['prediction'] == 1).sum()
        print(f"\nResults:")
        print(f"  Total transactions: {len(results)}")
        print(f"  Predicted fraud: {fraud_detected}")
        print(f"  Fraud rate: {fraud_detected/len(results)*100:.2f}%")
    
    # Save predictions
    results.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")
    
    return results


def predict_sample_transactions(model_path, data_path='creditcard.csv', n_samples=10):
    """
    Make predictions on random sample transactions and show details.
    
    Args:
        model_path: Path to saved model
        data_path: Path to original dataset
        n_samples: Number of samples to predict
    """
    print("Loading model...")
    model = FraudDetectionModel()
    model.load_model(model_path)
    
    print(f"Loading sample data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Sample random transactions
    sample = df.sample(n=n_samples, random_state=42)
    X = sample.drop('Class', axis=1)
    y_true = sample['Class']
    
    print(f"\nMaking predictions on {n_samples} sample transactions...")
    print("="*80)
    
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    for i, (idx, row) in enumerate(X.iterrows()):
        actual = "FRAUD" if y_true.iloc[i] == 1 else "NORMAL"
        predicted = "FRAUD" if predictions[i] == 1 else "NORMAL"
        prob = probabilities[i]
        correct = "✓" if predictions[i] == y_true.iloc[i] else "✗"
        
        print(f"\nTransaction #{i+1} (ID: {idx})")
        print(f"  Actual:     {actual}")
        print(f"  Predicted:  {predicted} {correct}")
        print(f"  Fraud Probability: {prob:.4f}")
        print(f"  Amount: ${row['Amount']:.2f}")


def predict_high_risk_transactions(model_path, data_path='creditcard.csv', 
                                   threshold=0.5, top_n=20):
    """
    Find and display highest risk transactions.
    
    Args:
        model_path: Path to saved model
        data_path: Path to dataset
        threshold: Probability threshold for fraud
        top_n: Number of top risky transactions to show
    """
    print("Loading model...")
    model = FraudDetectionModel()
    model.load_model(model_path)
    
    print(f"Analyzing transactions from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Get subset for analysis (use test portion)
    test_df = df.sample(frac=0.3, random_state=42)
    X = test_df.drop('Class', axis=1)
    y_true = test_df['Class']
    
    print(f"Scoring {len(X)} transactions...")
    probabilities = model.predict_proba(X)
    
    # Create results
    results = pd.DataFrame({
        'transaction_id': test_df.index,
        'fraud_probability': probabilities,
        'actual_fraud': y_true.values,
        'amount': test_df['Amount'].values
    })
    
    # Sort by probability
    results = results.sort_values('fraud_probability', ascending=False)
    
    # Filter high risk
    high_risk = results[results['fraud_probability'] >= threshold]
    
    print(f"\n{'='*80}")
    print(f"HIGH RISK TRANSACTIONS (Probability >= {threshold})")
    print(f"{'='*80}")
    print(f"Total flagged: {len(high_risk)}")
    print(f"Actually fraud: {high_risk['actual_fraud'].sum()}")
    print(f"False positives: {(high_risk['actual_fraud'] == 0).sum()}")
    
    print(f"\n{'='*80}")
    print(f"TOP {top_n} RISKIEST TRANSACTIONS")
    print(f"{'='*80}")
    print(f"{'ID':<10} {'Fraud Prob':>12} {'Amount':>12} {'Actual':>10}")
    print("-"*80)
    
    for _, row in results.head(top_n).iterrows():
        actual = "FRAUD" if row['actual_fraud'] == 1 else "NORMAL"
        print(f"{row['transaction_id']:<10} {row['fraud_probability']:>12.4f} "
              f"${row['amount']:>11.2f} {actual:>10}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make fraud predictions')
    parser.add_argument('--model-path', type=str, default='/data/models/best_fraud_detector.pkl',
                       help='Path to trained model')
    parser.add_argument('--data', type=str, default='/data/creditcard.csv',
                       help='Path to data file')
    parser.add_argument('--mode', type=str, default='sample',
                       choices=['sample', 'file', 'high-risk'],
                       help='Prediction mode')
    parser.add_argument('--input', type=str,
                       help='Input CSV file for file mode')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Output file for predictions')
    parser.add_argument('--samples', type=int, default=10,
                       help='Number of samples to predict')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Probability threshold for high-risk mode')
    parser.add_argument('--top', type=int, default=20,
                       help='Number of top risky transactions to show')
    
    args = parser.parse_args()
    
    if args.mode == 'sample':
        # Predict on random samples
        predict_sample_transactions(args.model_path, args.data, args.samples)
    elif args.mode == 'file':
        # Predict on entire file
        if not args.input:
            print("Error: --input required for file mode")
            exit(1)
        predict_from_csv(args.model_path, args.input, args.output)
    elif args.mode == 'high-risk':
        # Find high risk transactions
        predict_high_risk_transactions(args.model_path, args.data, 
                                      args.threshold, args.top)
