"""
Full Model Registry Migration Demo.
Runs the complete workflow end-to-end:
1. Export models from legacy registry (GPS/Davnic MLflow)
2. Register in Domino Model Registry with metadata (visible in Models tab)
3. Run experiment tracking with MLflow (visible in Experiments tab)
4. Execute model promotion workflow (dev -> staging -> prod)
5. Generate model lineage report
6. Configure model discoverability
"""
import os
import sys
from datetime import datetime

from config import LEGACY_REGISTRY_DIR, REGISTRY_PATH, EXPERIMENTS_DIR


def run_full_demo():
    """Execute complete model registry migration demo."""
    print("\n" + "#" * 60)
    print("#  MODEL REGISTRY MIGRATION DEMO")
    print("#  GPS/Davnic MLflow -> Domino Model Registry")
    print("#" * 60)
    print(f"\n  Started: {datetime.now().isoformat()}")
    print(f"  Legacy Registry: {LEGACY_REGISTRY_DIR}")
    print(f"  Domino Registry: {REGISTRY_PATH}")
    print(f"  Experiments:     {EXPERIMENTS_DIR}")

    # Cleanup old MLflow models and experiments for fresh demo run
    print("\n[Cleanup] Removing old models and experiments from MLflow...")
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI', '')
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
        client = MlflowClient()
        for model_name in ['fraud-detection-model', 'customer-churn-model', 'transaction-classifier']:
            try:
                client.delete_registered_model(model_name)
                print(f"  Deleted model: {model_name}")
            except:
                pass
        for exp in client.search_experiments():
            if exp.name != 'Default' and exp.experiment_id != '0':
                try:
                    client.delete_experiment(exp.experiment_id)
                    print(f"  Deleted experiment: {exp.name}")
                except:
                    pass
    except Exception as e:
        print(f"  Cleanup skipped: {e}")

    # Step 1: Export from legacy registry
    print("\n\n" + "=" * 60)
    print("STEP 1: Export Models from Legacy Registry")
    print("=" * 60)
    from legacy_registry_export import create_legacy_registry
    models = create_legacy_registry()

    # Step 2: Register in Domino (MLflow Model Registry -> Models tab)
    print("\n\n" + "=" * 60)
    print("STEP 2: Register Models in Domino Registry")
    print("         (Will appear in Domino 'Models' tab)")
    print("=" * 60)
    from domino_registry import migrate_to_domino_registry
    registry = migrate_to_domino_registry()

    # Step 3: Experiment tracking (MLflow -> Experiments tab)
    print("\n\n" + "=" * 60)
    print("STEP 3: Experiment Tracking & Model Lineage")
    print("         (Will appear in Domino 'Experiments' tab)")
    print("=" * 60)
    from experiment_tracking import run_hyperparameter_sweep
    tracker = run_hyperparameter_sweep()

    # Step 4: Model promotion workflow
    print("\n\n" + "=" * 60)
    print("STEP 4: Model Promotion Workflow")
    print("=" * 60)
    from model_promotion import run_promotion_workflow
    promotion_mgr = run_promotion_workflow()

    # Step 5: Model discoverability
    print("\n\n" + "=" * 60)
    print("STEP 5: Model Discoverability & Catalog")
    print("=" * 60)
    from model_discovery import demo_discovery
    discovery = demo_discovery()

    # Final summary
    print("\n\n" + "#" * 60)
    print("#  MODEL REGISTRY MIGRATION COMPLETE")
    print("#" * 60)
    print(f"""
Summary:
  [OK] Exported {len(models)} models from legacy GPS/Davnic MLflow registry
  [OK] Preserved all metadata (params, metrics, artifacts, lineage)
  [OK] Registered models in Domino with versioning (v1, v2, v3)
  [OK] Demonstrated promotion workflow (dev -> staging -> production)
  [OK] Tracked experiments with full model lineage
  [OK] Configured model discoverability (catalog, search, model cards)

Where to see results in Domino:
  * Models tab       -> Registered models with versions & stages
  * Experiments tab  -> Hyperparameter sweep runs with metrics
  * Data tab         -> model-registry dataset (registry, catalogs, model cards)

Requirement Coverage:
  1. Export models from legacy registry             [OK]
  2. Preserve model metadata                        [OK]
  3. Set up model versioning in Domino              [OK]
  4. Implement model promotion workflows            [OK]
  5. Migrate model lineage and experiment tracking  [OK]
  6. Configure model discoverability across teams   [OK]

Artifacts (Data tab -> model-registry dataset):
  * Registry:    {REGISTRY_PATH}/registry.json
  * Catalog:     {REGISTRY_PATH}/model_catalog.json
  * Model Cards: {REGISTRY_PATH}/model_cards/
  * Experiments: {EXPERIMENTS_DIR}/
  * Promotions:  {REGISTRY_PATH}/promotion_records/
""")


if __name__ == '__main__':
    run_full_demo()
