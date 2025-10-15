"""
Machine Learning Classifier for VAD
Author: Kiro Chen (530337094)
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

class VADClassifier:
    """
    Machine learning classifier for vocal activity detection
    """

    def __init__(self, model_type='random_forest', **kwargs):
        """
        Args:
            model_type: 'random_forest' or 'svm'
            **kwargs: Additional parameters for the model
        """
        self.model_type = model_type
        self.scaler = StandardScaler()

        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=42
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel=kwargs.get('kernel', 'rbf'),
                C=kwargs.get('C', 1.0),
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.is_trained = False

    def train(self, X, y):
        """
        Train the classifier

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,) - 0=instrumental, 1=vocal
        """
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True

        print(f"✓ Model trained on {X.shape[0]} samples")
        print(f"  Feature dimension: {X.shape[1]}")
        print(f"  Vocal frames: {np.sum(y)}")
        print(f"  Instrumental frames: {len(y) - np.sum(y)}")

    def predict(self, X, return_proba=False):
        """
        Predict vocal activity

        Args:
            X: Feature matrix (n_samples, n_features)
            return_proba: If True, return probabilities

        Returns:
            predictions: Binary predictions or probabilities
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet!")

        # Normalize
        X_scaled = self.scaler.transform(X)

        if return_proba:
            # Return probability of vocal class
            proba = self.model.predict_proba(X_scaled)
            return proba[:, 1]  # Probability of class 1 (vocal)
        else:
            return self.model.predict(X_scaled)

    def evaluate(self, X, y):
        """
        Evaluate model performance

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            metrics: Dictionary of evaluation metrics
        """
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        # Predict
        y_pred = self.predict(X)

        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred, average='binary', pos_label=1
        )

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        return metrics

    def save(self, filepath):
        """Save model to disk"""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type
        }
        joblib.dump(model_data, filepath)
        print(f"✓ Model saved to {filepath}")

    def load(self, filepath):
        """Load model from disk"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.is_trained = True
        print(f"✓ Model loaded from {filepath}")


# Test the classifier
if __name__ == "__main__":
    print("Testing VAD Classifier...")

    # Generate synthetic training data
    np.random.seed(42)

    # Simulate features for vocal frames (higher energy, different spectrum)
    n_vocal = 500
    X_vocal = np.random.randn(n_vocal, 50)  # 50 features
    X_vocal[:, 0] += 2.0  # Higher energy
    X_vocal[:, 1:13] += 1.0  # Different MFCCs
    y_vocal = np.ones(n_vocal)

    # Simulate features for instrumental frames
    n_inst = 500
    X_inst = np.random.randn(n_inst, 50)
    X_inst[:, 0] -= 0.5  # Lower energy
    y_inst = np.zeros(n_inst)

    # Combine
    X = np.vstack([X_vocal, X_inst])
    y = np.concatenate([y_vocal, y_inst])

    # Shuffle
    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Random Forest
    print("\n--- Random Forest ---")
    rf_clf = VADClassifier(model_type='random_forest', n_estimators=100)
    rf_clf.train(X_train, y_train)

    metrics_rf = rf_clf.evaluate(X_test, y_test)
    print(f"  Accuracy:  {metrics_rf['accuracy']:.3f}")
    print(f"  Precision: {metrics_rf['precision']:.3f}")
    print(f"  Recall:    {metrics_rf['recall']:.3f}")
    print(f"  F1-Score:  {metrics_rf['f1_score']:.3f}")

    # Train SVM
    print("\n--- SVM ---")
    svm_clf = VADClassifier(model_type='svm', C=1.0)
    svm_clf.train(X_train, y_train)

    metrics_svm = svm_clf.evaluate(X_test, y_test)
    print(f"  Accuracy:  {metrics_svm['accuracy']:.3f}")
    print(f"  Precision: {metrics_svm['precision']:.3f}")
    print(f"  Recall:    {metrics_svm['recall']:.3f}")
    print(f"  F1-Score:  {metrics_svm['f1_score']:.3f}")