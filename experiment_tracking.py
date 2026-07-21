"""
Experiment Tracking and Model Lineage.
Tracks all training runs, hyperparameter sweeps, and model lineage
using Domino Experiments API and local logging.
"""
import os
import json
import time
import numpy as np
from datetime import datetime
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import requests

from config import (
    DOMINO_API_HOST, DOMINO_API_KEY, DOMINO_RUN_ID,
    EXPERIMENTS_DIR, RANDOM_SEED
)


class ExperimentTracker:
    """Track experiments and model lineage in Domino."""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join(EXPERIMENTS_DIR, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        self.runs = self._load_runs()

    def _load_runs(self) -> list:
        """Load existing experiment runs."""
        runs_file = os.path.join(self.experiment_dir, 'runs.json')
        if os.path.exists(runs_file):
            with open(runs_file, 'r') as f:
                return json.load(f)
        return []

    def _save_runs(self):
        """Save experiment runs."""
        runs_file = os.path.join(self.experiment_dir, 'runs.json')
        with open(runs_file, 'w') as f:
            json.dump(self.runs, f, indent=2)

    def start_run(self, run_name: str, tags: dict = None) -> dict:
        """Start a new experiment run."""
        run = {
            'run_id': f"run_{len(self.runs) + 1:03d}",
            'run_name': run_name,
            'experiment': self.experiment_name,
            'status': 'running',
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'parameters': {},
            'metrics': {},
            'tags': tags or {},
            'artifacts': [],
            'domino_run_id': DOMINO_RUN_ID,
        }
        self.runs.append(run)
        return run

    def log_params(self, run_id: str, params: dict):
        """Log hyperparameters for a run."""
        for run in self.runs:
            if run['run_id'] == run_id:
                run['parameters'].update(params)
                break
        self._save_runs()

        # Log to Domino stats API
        self._log_to_domino_stats(params, prefix='param')

    def log_metrics(self, run_id: str, metrics: dict):
        """Log metrics for a run."""
        for run in self.runs:
            if run['run_id'] == run_id:
                run['metrics'].update(metrics)
                break
        self._save_runs()

        # Log to Domino stats API
        self._log_to_domino_stats(metrics, prefix='metric')

    def end_run(self, run_id: str, status: str = 'completed'):
        """End an experiment run."""
        for run in self.runs:
            if run['run_id'] == run_id:
                run['status'] = status
                run['end_time'] = datetime.now().isoformat()
                break
        self._save_runs()

    def _log_to_domino_stats(self, data: dict, prefix: str = ''):
        """Log stats to Domino's experiment tracking via API."""
        if not DOMINO_API_HOST or not DOMINO_API_KEY:
            return

        url = f"{DOMINO_API_HOST}/v1/runs/{DOMINO_RUN_ID}/stats"
        stats = {f"{prefix}_{k}" if prefix else k: v for k, v in data.items()}

        try:
            response = requests.post(
                url, headers={'X-Domino-Api-Key': DOMINO_API_KEY},
                json=stats, timeout=10
            )
        except Exception:
            pass

    def get_best_run(self, metric: str = 'accuracy', higher_is_better: bool = True) -> dict:
        """Get the best run based on a metric."""
        completed = [r for r in self.runs if r['status'] == 'completed']
        if not completed:
            return {}

        if higher_is_better:
            return max(completed, key=lambda r: r['metrics'].get(metric, 0))
        return min(completed, key=lambda r: r['metrics'].get(metric, float('inf')))

    def get_lineage(self) -> dict:
        """Get full model lineage (experiment → runs → best model)."""
        best = self.get_best_run()
        return {
            'experiment': self.experiment_name,
            'total_runs': len(self.runs),
            'completed_runs': len([r for r in self.runs if r['status'] == 'completed']),
            'best_run': best.get('run_id', 'N/A'),
            'best_metrics': best.get('metrics', {}),
            'best_params': best.get('parameters', {}),
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
        completed = [r for r in self.runs if r['status'] == 'completed']
        if completed:
            best = self.get_best_run()
            print(f"  Best run: {best['run_id']} (accuracy={best['metrics'].get('accuracy', 'N/A')})")
            print(f"  Best params: {best['parameters']}")


def run_hyperparameter_sweep():
    """
    Run a hyperparameter sweep to demonstrate experiment tracking.
    Simulates what would happen during model development.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT TRACKING: Hyperparameter Sweep")
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
        {'n_estimators': 300, 'max_depth': None, 'min_samples_split': 2},
    ]

    print(f"\n  Running {len(param_grid)} experiments...")

    for i, params in enumerate(param_grid):
        run = tracker.start_run(
            f"sweep_run_{i+1}",
            tags={'sweep_id': 'hp_sweep_001', 'iteration': i + 1}
        )

        tracker.log_params(run['run_id'], params)

        model = RandomForestClassifier(random_state=RANDOM_SEED, **params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': round(accuracy_score(y_test, y_pred), 4),
            'f1_score': round(f1_score(y_test, y_pred), 4),
            'roc_auc': round(roc_auc_score(y_test, y_proba), 4),
        }
        tracker.log_metrics(run['run_id'], metrics)
        tracker.end_run(run['run_id'])

        print(f"    Run {i+1}: n_est={params['n_estimators']:3d}, "
              f"depth={str(params['max_depth']):4s} → "
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
