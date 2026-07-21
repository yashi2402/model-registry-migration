"""
Legacy Registry Export.
Simulates exporting models from a legacy MLflow/custom registry.
Creates a realistic legacy registry structure with models, metadata, and artifacts.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.datasets import make_classification, load_iris
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from config import LEGACY_REGISTRY_DIR, RANDOM_SEED


def create_legacy_registry():
    """
    Simulate a legacy MLflow-style registry with multiple models,
    versions, and experiment runs.
    """
    os.makedirs(LEGACY_REGISTRY_DIR, exist_ok=True)
    print("=" * 60)
    print("EXPORTING MODELS FROM LEGACY REGISTRY (GPS/Davnic MLflow)")
    print("=" * 60)

    models_exported = []

    # Model 1: Fraud Detection (Random Forest)
    print("\n[1/3] Exporting: fraud-detection-model")
    model_1 = train_fraud_detection_model()
    models_exported.append(model_1)

    # Model 2: Customer Churn (Gradient Boosting)
    print("\n[2/3] Exporting: customer-churn-model")
    model_2 = train_churn_model()
    models_exported.append(model_2)

    # Model 3: Transaction Classification (Logistic Regression)
    print("\n[3/3] Exporting: transaction-classifier")
    model_3 = train_transaction_classifier()
    models_exported.append(model_3)

    # Save registry catalog
    catalog = {
        'registry_source': 'GPS/Davnic MLflow Registry',
        'export_time': datetime.now().isoformat(),
        'total_models': len(models_exported),
        'models': [m['metadata']['model_name'] for m in models_exported],
        'total_versions': sum(m['metadata']['version'] for m in models_exported),
    }

    catalog_path = os.path.join(LEGACY_REGISTRY_DIR, 'registry_catalog.json')
    with open(catalog_path, 'w') as f:
        json.dump(catalog, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Legacy registry exported: {len(models_exported)} models")
    print(f"Location: {LEGACY_REGISTRY_DIR}")
    print(f"{'=' * 60}")

    return models_exported


def train_fraud_detection_model():
    """Train fraud detection model with multiple versions."""
    model_dir = os.path.join(LEGACY_REGISTRY_DIR, 'fraud-detection-model')
    os.makedirs(model_dir, exist_ok=True)

    X, y = make_classification(
        n_samples=10000, n_features=20, n_informative=12,
        n_redundant=4, weights=[0.95, 0.05],
        random_state=RANDOM_SEED
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Version 1: Basic model
    model_v1 = RandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED)
    model_v1.fit(X_train_scaled, y_train)
    y_pred_v1 = model_v1.predict(X_test_scaled)
    y_proba_v1 = model_v1.predict_proba(X_test_scaled)[:, 1]

    # Version 2: Improved model
    model_v2 = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        random_state=RANDOM_SEED
    )
    model_v2.fit(X_train_scaled, y_train)
    y_pred_v2 = model_v2.predict(X_test_scaled)
    y_proba_v2 = model_v2.predict_proba(X_test_scaled)[:, 1]

    # Save version 2 (latest/best)
    model_path = os.path.join(model_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_v2, f)

    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    metadata = {
        'model_name': 'fraud-detection-model',
        'model_type': 'RandomForestClassifier',
        'version': 2,
        'framework': 'scikit-learn',
        'created_by': 'data_science_team',
        'created_at': (datetime.now() - timedelta(days=90)).isoformat(),
        'updated_at': (datetime.now() - timedelta(days=15)).isoformat(),
        'description': 'Detects fraudulent transactions in real-time payment processing',
        'use_case': 'fraud_detection',
        'stage': 'production',
        'parameters': {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'random_state': RANDOM_SEED,
        },
        'metrics': {
            'accuracy': round(accuracy_score(y_test, y_pred_v2), 4),
            'precision': round(precision_score(y_test, y_pred_v2), 4),
            'recall': round(recall_score(y_test, y_pred_v2), 4),
            'f1_score': round(f1_score(y_test, y_pred_v2), 4),
            'roc_auc': round(roc_auc_score(y_test, y_proba_v2), 4),
        },
        'training_data': {
            'n_samples': 10000,
            'n_features': 20,
            'class_balance': '95% legit / 5% fraud',
            'source': 'GPS/Davnic transaction database',
        },
        'artifacts': ['model.pkl', 'scaler.pkl', 'feature_importance.json'],
        'version_history': [
            {'version': 1, 'accuracy': round(accuracy_score(y_test, y_pred_v1), 4), 'stage': 'archived'},
            {'version': 2, 'accuracy': round(accuracy_score(y_test, y_pred_v2), 4), 'stage': 'production'},
        ]
    }

    # Feature importance
    importance = dict(zip(
        [f'feature_{i}' for i in range(20)],
        [round(float(x), 4) for x in model_v2.feature_importances_]
    ))
    with open(os.path.join(model_dir, 'feature_importance.json'), 'w') as f:
        json.dump(importance, f, indent=2)

    with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Exported: {model_dir}")
    print(f"  Metrics: accuracy={metadata['metrics']['accuracy']}, auc={metadata['metrics']['roc_auc']}")

    return {'metadata': metadata, 'model': model_v2, 'scaler': scaler}


def train_churn_model():
    """Train customer churn prediction model."""
    model_dir = os.path.join(LEGACY_REGISTRY_DIR, 'customer-churn-model')
    os.makedirs(model_dir, exist_ok=True)

    X, y = make_classification(
        n_samples=8000, n_features=15, n_informative=10,
        n_redundant=3, weights=[0.75, 0.25],
        random_state=RANDOM_SEED + 1
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    model = GradientBoostingClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.1,
        random_state=RANDOM_SEED
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    model_path = os.path.join(model_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    metadata = {
        'model_name': 'customer-churn-model',
        'model_type': 'GradientBoostingClassifier',
        'version': 3,
        'framework': 'scikit-learn',
        'created_by': 'analytics_team',
        'created_at': (datetime.now() - timedelta(days=180)).isoformat(),
        'updated_at': (datetime.now() - timedelta(days=30)).isoformat(),
        'description': 'Predicts customer churn probability for retention campaigns',
        'use_case': 'churn_prediction',
        'stage': 'staging',
        'parameters': {
            'n_estimators': 150,
            'max_depth': 6,
            'learning_rate': 0.1,
        },
        'metrics': {
            'accuracy': round(accuracy_score(y_test, y_pred), 4),
            'precision': round(precision_score(y_test, y_pred), 4),
            'recall': round(recall_score(y_test, y_pred), 4),
            'f1_score': round(f1_score(y_test, y_pred), 4),
            'roc_auc': round(roc_auc_score(y_test, y_proba), 4),
        },
        'training_data': {
            'n_samples': 8000,
            'n_features': 15,
            'class_balance': '75% active / 25% churned',
            'source': 'GPS/Davnic CRM database',
        },
        'artifacts': ['model.pkl'],
        'version_history': [
            {'version': 1, 'accuracy': 0.82, 'stage': 'archived'},
            {'version': 2, 'accuracy': 0.87, 'stage': 'archived'},
            {'version': 3, 'accuracy': round(accuracy_score(y_test, y_pred), 4), 'stage': 'staging'},
        ]
    }

    with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Exported: {model_dir}")
    print(f"  Metrics: accuracy={metadata['metrics']['accuracy']}, auc={metadata['metrics']['roc_auc']}")

    return {'metadata': metadata, 'model': model}


def train_transaction_classifier():
    """Train transaction type classifier."""
    model_dir = os.path.join(LEGACY_REGISTRY_DIR, 'transaction-classifier')
    os.makedirs(model_dir, exist_ok=True)

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=RANDOM_SEED
    )

    model = LogisticRegression(max_iter=200, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cv_scores = cross_val_score(model, iris.data, iris.target, cv=5)

    model_path = os.path.join(model_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    metadata = {
        'model_name': 'transaction-classifier',
        'model_type': 'LogisticRegression',
        'version': 1,
        'framework': 'scikit-learn',
        'created_by': 'ml_engineering',
        'created_at': (datetime.now() - timedelta(days=60)).isoformat(),
        'updated_at': (datetime.now() - timedelta(days=60)).isoformat(),
        'description': 'Classifies transactions into categories for routing',
        'use_case': 'transaction_classification',
        'stage': 'development',
        'parameters': {
            'max_iter': 200,
            'solver': 'lbfgs',
        },
        'metrics': {
            'accuracy': round(accuracy_score(y_test, y_pred), 4),
            'cv_mean_accuracy': round(float(cv_scores.mean()), 4),
            'cv_std': round(float(cv_scores.std()), 4),
        },
        'training_data': {
            'n_samples': 150,
            'n_features': 4,
            'n_classes': 3,
            'source': 'GPS/Davnic transaction logs',
        },
        'artifacts': ['model.pkl'],
        'version_history': [
            {'version': 1, 'accuracy': round(accuracy_score(y_test, y_pred), 4), 'stage': 'development'},
        ]
    }

    with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Exported: {model_dir}")
    print(f"  Metrics: accuracy={metadata['metrics']['accuracy']}, cv_mean={metadata['metrics']['cv_mean_accuracy']}")

    return {'metadata': metadata, 'model': model}


if __name__ == '__main__':
    create_legacy_registry()
