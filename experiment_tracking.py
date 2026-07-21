"""
Experiment Tracking and Model Lineage.
Tracks all training runs, hyperparameter sweeps, and model lineage
using MLflow (integrated with Domino Experiments tab) and local logging.
"""
import os
import json
import numpy as np
from datetime import datetime
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from config import EXPERIMENTS_DIR, RANDOM_SEED, MLFLOW_TRACKING_URI


class ExperimentTracker:
    """Track experiments using MLflow (Domino Experiments tab) + local JSON backup."""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join(EXPERIMENTS_DIR, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        self.runs = []
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Initialize MLflow tracking for Domino Experiments tab."""
        self.mlflow_enabled = False
        if not MLFLOW_AVAILABLE:
            print("    [INFO] mlflow not installed - using local tracking only")
            return

        try:
            if MLFLOW_TRACKING_URI:
                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            self._set_experiment_safe(self.experiment_name)
            self.mlflow_enabled = True
            print(f"    [INFO] MLflow tracking enabled - experiments will appear in Domino Experiments tab")
        except Exception as e:
            print(f"    [INFO] MLflow setup skipped ({e}) - using local tracking")

    def _set_experiment_safe(self, name):
        """Set MLflow experiment, restoring it if it was previously deleted."""
        try:
            mlflow.set_experiment(name)
        except Exception:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            exp = client.get_experiment_by_name(name)
            if exp and exp.lifecycle_stage == 'deleted':
                client.restore_experiment(exp.experiment_id)
                mlflow.set_experiment(name)
            else:
                raise

    def run_experiment(self, run_name: str, model, X_train, y_train,
                       X_test, y_test, params: dict, tags: dict = None) -> dict:
        """
        Run a single experiment: train, evaluate, log to MLflow + local.
        This makes the run visible in Domino's Experiments tab.
        """
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': round(accuracy_score(y_test, y_pred), 4),
            'f1_score': round(f1_score(y_test, y_pred), 4),
            'roc_auc': round(roc_auc_score(y_test, y_proba), 4),
        }

        run_record = {
            'run_id': f"run_{len(self.runs) + 1:03d}",
            'run_name': run_name,
            'experiment': self.experiment_name,
            'status': 'completed',
            'start_time': datetime.now().isoformat(),
            'parameters': params,
            'metrics': metrics,
            'tags': tags or {},
        }

        # Log to MLflow (appears in Domino Experiments tab)
        if self.mlflow_enabled:
            try:
                with mlflow.start_run(run_name=run_name):
                    mlflow.log_params(params)
                    mlflow.log_metrics(metrics)
                    if tags:
                        mlflow.set_tags(tags)
                    mlflow.sklearn.log_model(model, "model")
                    run_record['mlflow_run_id'] = mlflow.active_run().info.run_id
            except Exception as e:
                print(f"    [WARN] MLflow logging failed: {e}")

        self.runs.append(run_record)
        self._save_runs()
        return run_record

    def _save_runs(self):
        """Save experiment runs to local JSON (backup)."""
        runs_file = os.path.join(self.experiment_dir, 'runs.json')
        with open(runs_file, 'w') as f:
            json.dump(self.runs, f, indent=2)

    def get_best_run(self, metric: str = 'accuracy', higher_is_better: bool = True) -> dict:
        """Get the best run based on a metric."""
        completed = [r for r in self.runs if r['status'] == 'completed']
        if not completed:
            return {}
        if higher_is_better:
            return max(completed, key=lambda r: r['metrics'].get(metric, 0))
        return min(completed, key=lambda r: r['metrics'].get(metric, float('inf')))

    def get_lineage(self) -> dict:
        """Get full model lineage (experiment -> runs -> best model)."""
        best = self.get_best_run()
        return {
            'experiment': self.experiment_name,
            'total_runs': len(self.runs),
            'completed_runs': len([r for r in self.runs if r['status'] == 'completed']),
            'best_run': best.get('run_id', 'N/A'),
            'best_metrics': best.get('metrics', {}),
            'best_params': best.get('parameters', {}),
            'mlflow_enabled': self.mlflow_enabled,
            'run_history': [
                {
                    'run_id': r['run_id'],
                    'accuracy': r['metrics'].get('accuracy', 'N/A'),
                    'status': r['status'],
                    'time': r['start_time'],
                }
                for r in self.runs
            ]
        }

    def summary(self):
        """Print experiment summary."""
        print(f"\n  Experiment: {self.experiment_name}")
        print(f"  Total runs: {len(self.runs)}")
        print(f"  MLflow tracking: {'ENABLED (visible in Experiments tab)' if self.mlflow_enabled else 'LOCAL ONLY'}")
        completed = [r for r in self.runs if r['status'] == 'completed']
        if completed:
            best = self.get_best_run()
            print(f"  Best run: {best['run_id']} (accuracy={best['metrics'].get('accuracy', 'N/A')})")
            print(f"  Best params: {best['parameters']}")


def run_hyperparameter_sweep():
    """
    Run a hyperparameter sweep to demonstrate experiment tracking.
    Results appear in Domino's Experiments tab via MLflow.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT TRACKING: Hyperparameter Sweep")
    print("(Results will appear in Domino Experiments tab)")
    print("=" * 60)

    X, y = make_classification(
        n_samples=5000, n_features=15, n_informative=10,
        random_state=RANDOM_SEED
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    tracker = ExperimentTracker('fraud-detection-sweep')

    param_grid = [
        {'n_estimators': 50, 'max_depth': 5, 'min_samples_split': 2},
        {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5},
        {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 3},
        {'n_estimators': 150, 'max_depth': 12, 'min_samples_split': 4},
        {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 2},
    ]

    print(f"\n  Running {len(param_grid)} experiments...")

    for i, params in enumerate(param_grid):
        model = RandomForestClassifier(random_state=RANDOM_SEED, **params)
        run = tracker.run_experiment(
            run_name=f"sweep_run_{i+1}",
            model=model,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            params={k: str(v) for k, v in params.items()},
            tags={'sweep_id': 'hp_sweep_001', 'iteration': str(i + 1)}
        )

        metrics = run['metrics']
        print(f"    Run {i+1}: n_est={params['n_estimators']:3d}, "
              f"depth={str(params['max_depth']):4s} -> "
              f"accuracy={metrics['accuracy']:.4f}, auc={metrics['roc_auc']:.4f}")

    tracker.summary()

    # Save lineage
    lineage = tracker.get_lineage()
    lineage_path = os.path.join(EXPERIMENTS_DIR, 'fraud-detection-sweep', 'lineage.json')
    with open(lineage_path, 'w') as f:
        json.dump(lineage, f, indent=2)

    print(f"\n  Lineage saved: {lineage_path}")
    print(f"{'=' * 60}")

    return tracker


if __name__ == '__main__':
    run_hyperparameter_sweep()
