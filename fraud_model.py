"""
Credit Card Fraud Detection Model
Handles highly imbalanced classification with multiple algorithms
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_curve, roc_auc_score, 
    f1_score, precision_score, recall_score,
    average_precision_score
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
warnings.filterwarnings('ignore')


class FraudDetectionModel:
    """
    Credit Card Fraud Detection with multiple algorithms and imbalanced data handling.
    """
    
    def __init__(self, model_type='logistic', sampling_strategy='smote'):
        """
        Initialize the fraud detection model.
        
        Args:
            model_type: 'logistic', 'random_forest', 'weighted_logistic'
            sampling_strategy: 'smote', 'undersample', 'none'
        """
        self.model_type = model_type
        self.sampling_strategy = sampling_strategy
        self.model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self, filepath='creditcard.csv', test_size=0.3, random_state=42):
        """
        Load and split the credit card fraud dataset.
        
        Args:
            filepath: Path to creditcard.csv
            test_size: Fraction for test set
            random_state: Random seed
        """
        print("Loading data...")
        df = pd.read_csv(filepath)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].sum()/len(df)*100:.3f}%)")
        print(f"Normal cases: {(df['Class']==0).sum()} ({(df['Class']==0).sum()/len(df)*100:.3f}%)")
        
        # Separate features and target
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nTraining set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        print(f"Fraud in train: {self.y_train.sum()} ({self.y_train.sum()/len(self.y_train)*100:.3f}%)")
        print(f"Fraud in test: {self.y_test.sum()} ({self.y_test.sum()/len(self.y_test)*100:.3f}%)")
        
        return self
    
    def preprocess(self):
        """
        Scale features using StandardScaler.
        """
        print("\nScaling features...")
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        return self
    
    def build_model(self):
        """
        Build the classification model with chosen algorithm and sampling strategy.
        """
        print(f"\nBuilding {self.model_type} model with {self.sampling_strategy} sampling...")
        
        # Choose base model
        if self.model_type == 'logistic':
            base_model = LogisticRegression(max_iter=1000, random_state=42)
        elif self.model_type == 'weighted_logistic':
            # Use class weights to handle imbalance
            base_model = LogisticRegression(
                max_iter=1000, 
                class_weight='balanced',
                random_state=42
            )
        elif self.model_type == 'random_forest':
            base_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Apply sampling strategy
        if self.sampling_strategy == 'smote':
            # SMOTE oversampling
            sampler = SMOTE(random_state=42)
            self.model = ImbPipeline([
                ('sampler', sampler),
                ('classifier', base_model)
            ])
        elif self.sampling_strategy == 'undersample':
            # Random undersampling
            sampler = RandomUnderSampler(random_state=42)
            self.model = ImbPipeline([
                ('sampler', sampler),
                ('classifier', base_model)
            ])
        else:
            # No sampling
            self.model = base_model
        
        return self
    
    def train(self):
        """
        Train the fraud detection model.
        """
        print("\nTraining model...")
        self.model.fit(self.X_train, self.y_train)
        print("Training completed!")
        return self
    
    def evaluate(self, verbose=True):
        """
        Evaluate the model on test set with comprehensive metrics.
        
        Args:
            verbose: Print detailed metrics
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("\n" + "="*80)
        print("MODEL EVALUATION")
        print("="*80)
        
        # Predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': (y_pred == self.y_test).mean(),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'avg_precision': average_precision_score(self.y_test, y_pred_proba)
        }
        
        if verbose:
            print(f"\nAccuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
            print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"Average Precision: {metrics['avg_precision']:.4f}")
            
            print("\n" + "-"*80)
            print("CONFUSION MATRIX")
            print("-"*80)
            cm = confusion_matrix(self.y_test, y_pred)
            print(f"True Negatives:  {cm[0,0]:>6}")
            print(f"False Positives: {cm[0,1]:>6}")
            print(f"False Negatives: {cm[1,0]:>6}")
            print(f"True Positives:  {cm[1,1]:>6}")
            
            print("\n" + "-"*80)
            print("CLASSIFICATION REPORT")
            print("-"*80)
            print(classification_report(self.y_test, y_pred, 
                                      target_names=['Normal', 'Fraud']))
        
        return metrics
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix (unscaled)
            
        Returns:
            Predictions (0 or 1)
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Get fraud probabilities for new data.
        
        Args:
            X: Feature matrix (unscaled)
            
        Returns:
            Array of fraud probabilities
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def save_model(self, filepath='models/fraud_detector.pkl'):
        """
        Save the trained model and scaler.
        
        Args:
            filepath: Path to save model
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'sampling_strategy': self.sampling_strategy
        }
        
        joblib.dump(model_data, filepath)
        print(f"\nModel saved to {filepath}")
    
    def load_model(self, filepath='models/fraud_detector.pkl'):
        """
        Load a trained model.
        
        Args:
            filepath: Path to saved model
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.sampling_strategy = model_data['sampling_strategy']
        print(f"Model loaded from {filepath}")
        return self


def compare_models(filepath='creditcard.csv'):
    """
    Train and compare multiple models with different strategies.
    
    Args:
        filepath: Path to dataset
    """
    configs = [
        {'model_type': 'logistic', 'sampling_strategy': 'none', 'name': 'Logistic (No Sampling)'},
        {'model_type': 'weighted_logistic', 'sampling_strategy': 'none', 'name': 'Logistic (Weighted)'},
        {'model_type': 'logistic', 'sampling_strategy': 'smote', 'name': 'Logistic + SMOTE'},
        {'model_type': 'logistic', 'sampling_strategy': 'undersample', 'name': 'Logistic + Undersample'},
        {'model_type': 'random_forest', 'sampling_strategy': 'smote', 'name': 'Random Forest + SMOTE'},
    ]
    
    results = []
    
    for config in configs:
        print("\n" + "="*80)
        print(f"Training: {config['name']}")
        print("="*80)
        
        model = FraudDetectionModel(
            model_type=config['model_type'],
            sampling_strategy=config['sampling_strategy']
        )
        
        model.load_data(filepath)
        model.preprocess()
        model.build_model()
        model.train()
        
        metrics = model.evaluate(verbose=False)
        metrics['name'] = config['name']
        results.append(metrics)
        
        print(f"F1: {metrics['f1']:.4f} | Recall: {metrics['recall']:.4f} | Precision: {metrics['precision']:.4f}")
    
    # Print comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(f"{'Model':<30} {'F1':>8} {'Recall':>8} {'Precision':>8} {'ROC-AUC':>8}")
    print("-"*80)
    
    for result in results:
        print(f"{result['name']:<30} {result['f1']:>8.4f} {result['recall']:>8.4f} "
              f"{result['precision']:>8.4f} {result['roc_auc']:>8.4f}")
    
    # Find best model by F1
    best_model = max(results, key=lambda x: x['f1'])
    print("\n" + "="*80)
    print(f"BEST MODEL: {best_model['name']} (F1: {best_model['f1']:.4f})")
    print("="*80)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Credit Card Fraud Detection Model')
    parser.add_argument('--data', type=str, default='creditcard.csv',
                       help='Path to creditcard.csv')
    parser.add_argument('--model', type=str, default='logistic',
                       choices=['logistic', 'weighted_logistic', 'random_forest'],
                       help='Model type')
    parser.add_argument('--sampling', type=str, default='smote',
                       choices=['smote', 'undersample', 'none'],
                       help='Sampling strategy')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple models')
    parser.add_argument('--save', type=str, default='models/fraud_detector.pkl',
                       help='Path to save model')
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare all models
        compare_models(args.data)
    else:
        # Train single model
        detector = FraudDetectionModel(
            model_type=args.model,
            sampling_strategy=args.sampling
        )
        
        detector.load_data(args.data)
        detector.preprocess()
        detector.build_model()
        detector.train()
        detector.evaluate()
        detector.save_model(args.save)
