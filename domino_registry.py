"""
Domino Model Registry Integration.
Registers exported models into Domino's Model Registry using MLflow
(appears in Models tab) with full metadata, versioning, and promotion workflows.
"""
import os
import json
import pickle
import shutil
from datetime import datetime
from typing import Optional

try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from config import (
    DOMINO_API_HOST, DOMINO_API_KEY, DOMINO_PROJECT_ID,
    DOMINO_PROJECT_OWNER, DOMINO_PROJECT_NAME,
    LEGACY_REGISTRY_DIR, REGISTRY_PATH, PROMOTION_STAGES,
    MLFLOW_TRACKING_URI
)


class DominoModelRegistry:
    """Manage models in Domino's Model Registry via MLflow + local storage."""

    def __init__(self):
        self.api_host = DOMINO_API_HOST
        self.api_key = DOMINO_API_KEY
        self.project_id = DOMINO_PROJECT_ID
        self.registry = self._load_registry()
        self.mlflow_enabled = False
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Initialize MLflow for Domino Model Registry (Models tab)."""
        if not MLFLOW_AVAILABLE:
            print("    [INFO] mlflow not installed - models stored locally only")
            return
        try:
            if MLFLOW_TRACKING_URI:
                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            self.mlflow_client = MlflowClient()
            self.mlflow_enabled = True
            print("    [INFO] MLflow Model Registry enabled - models will appear in Domino Models tab")
        except Exception as e:
            print(f"    [INFO] MLflow Model Registry setup skipped ({e})")

    def _load_registry(self) -> dict:
        """Load or initialize the local model registry."""
        registry_file = os.path.join(REGISTRY_PATH, 'registry.json')
        if os.path.exists(registry_file):
            with open(registry_file, 'r') as f:
                return json.load(f)
        return {
            'registry_name': 'Domino Model Registry',
            'migrated_from': 'GPS/Davnic MLflow',
            'created_at': datetime.now().isoformat(),
            'models': {},
            'promotion_history': [],
        }

    def _save_registry(self):
        """Persist registry state locally."""
        os.makedirs(REGISTRY_PATH, exist_ok=True)
        registry_file = os.path.join(REGISTRY_PATH, 'registry.json')
        with open(registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def register_model(self, model_name: str, metadata: dict,
                       model_path: str, artifacts: list = None) -> dict:
        """
        Register a model in the Domino Model Registry.
        Uses MLflow to make it visible in the Models tab.
        Also stores locally for backup.
        """
        version = metadata.get('version', 1)
        model_key = f"{model_name}_v{version}"

        entry = {
            'model_name': model_name,
            'version': version,
            'registered_at': datetime.now().isoformat(),
            'migrated_from': 'GPS/Davnic MLflow Registry',
            'stage': metadata.get('stage', 'development'),
            'model_type': metadata.get('model_type', 'unknown'),
            'framework': metadata.get('framework', 'unknown'),
            'description': metadata.get('description', ''),
            'created_by': metadata.get('created_by', 'unknown'),
            'parameters': metadata.get('parameters', {}),
            'metrics': metadata.get('metrics', {}),
            'training_data': metadata.get('training_data', {}),
            'model_path': model_path,
            'artifacts': artifacts or [],
            'lineage': {
                'source_system': 'GPS/Davnic',
                'original_created': metadata.get('created_at', ''),
                'original_updated': metadata.get('updated_at', ''),
                'migration_time': datetime.now().isoformat(),
                'version_history': metadata.get('version_history', []),
            },
            'tags': [metadata.get('use_case', ''), metadata.get('framework', '')],
        }

        # Store in local registry
        if model_name not in self.registry['models']:
            self.registry['models'][model_name] = {'versions': {}}
        self.registry['models'][model_name]['versions'][str(version)] = entry
        self.registry['models'][model_name]['latest_version'] = version
        self.registry['models'][model_name]['current_stage'] = entry['stage']

        # Copy model artifact to local registry storage
        model_store = os.path.join(REGISTRY_PATH, model_name, f'v{version}')
        os.makedirs(model_store, exist_ok=True)
        if os.path.exists(model_path):
            shutil.copy2(model_path, os.path.join(model_store, 'model.pkl'))

        with open(os.path.join(model_store, 'metadata.json'), 'w') as f:
            json.dump(entry, f, indent=2)

        self._save_registry()

        print(f"  Registered: {model_name} v{version} [{entry['stage']}]")
        print(f"  Stored at:  {model_store}")

        # Register in MLflow Model Registry (appears in Domino Models tab)
        self._register_with_mlflow(model_name, model_path, metadata, entry)

        return entry

    def _set_experiment_safe(self, name):
        """Set MLflow experiment, restoring it if it was previously deleted."""
        try:
            mlflow.set_experiment(name)
        except Exception:
            client = MlflowClient()
            exp = client.get_experiment_by_name(name)
            if exp and exp.lifecycle_stage == 'deleted':
                client.restore_experiment(exp.experiment_id)
                mlflow.set_experiment(name)
            else:
                raise

    def _register_with_mlflow(self, model_name: str, model_path: str,
                               metadata: dict, entry: dict):
        """Register model with MLflow so it appears in Domino's Models tab.
        Registers ALL versions from version_history so multiple versions show up."""
        if not self.mlflow_enabled:
            return

        try:
            if not os.path.exists(model_path):
                print(f"    [WARN] Model file not found: {model_path}")
                return

            with open(model_path, 'rb') as f:
                model_obj = pickle.load(f)

            self._set_experiment_safe(f"migration-{model_name}")

            version_history = metadata.get('version_history', [])
            stage_map = {
                'production': 'Production',
                'staging': 'Staging',
                'archived': 'Archived',
                'development': 'None',
            }

            if not version_history:
                version_history = [{'version': metadata.get('version', 1), 'stage': metadata.get('stage', 'development')}]

            for vh in version_history:
                v_num = vh.get('version', 1)
                v_stage = vh.get('stage', 'development')

                with mlflow.start_run(run_name=f"migrate-{model_name}-v{v_num}"):
                    params_to_log = {}
                    for k, v in metadata.get('parameters', {}).items():
                        params_to_log[k] = str(v)
                    params_to_log['model_type'] = metadata.get('model_type', '')
                    params_to_log['framework'] = metadata.get('framework', '')
                    params_to_log['migrated_from'] = 'GPS/Davnic MLflow'
                    params_to_log['version'] = str(v_num)
                    mlflow.log_params(params_to_log)

                    if 'accuracy' in vh:
                        mlflow.log_metric('accuracy', vh['accuracy'])
                    for k, v in metadata.get('metrics', {}).items():
                        if isinstance(v, (int, float)) and k != 'accuracy':
                            mlflow.log_metric(k, v)

                    mlflow.set_tag('use_case', metadata.get('use_case', ''))
                    mlflow.set_tag('created_by', metadata.get('created_by', ''))
                    mlflow.set_tag('migration_source', 'GPS/Davnic')
                    mlflow.set_tag('version_stage', v_stage)
                    mlflow.set_tag('version_number', str(v_num))

                    mlflow.sklearn.log_model(
                        model_obj,
                        "model",
                        registered_model_name=model_name
                    )

                mlflow_stage = stage_map.get(v_stage, 'None')
                try:
                    all_versions = self.mlflow_client.search_model_versions(f"name='{model_name}'")
                    if all_versions:
                        latest_mv = max(all_versions, key=lambda x: int(x.version))
                        if mlflow_stage != 'None':
                            self.mlflow_client.transition_model_version_stage(
                                name=model_name,
                                version=latest_mv.version,
                                stage=mlflow_stage
                            )
                except Exception:
                    pass

                print(f"    [MLflow] Registered v{v_num} (stage: {v_stage})")

            print(f"    [MLflow] Total {len(version_history)} versions in Models tab")

        except Exception as e:
            print(f"    [MLflow] Registration skipped: {e}")

    def promote_model(self, model_name: str, version: int,
                      target_stage: str, reason: str = '') -> bool:
        """Promote a model to a new stage (dev -> staging -> production)."""
        if target_stage not in PROMOTION_STAGES:
            print(f"  ERROR: Invalid stage '{target_stage}'. Must be one of {PROMOTION_STAGES}")
            return False

        if model_name not in self.registry['models']:
            print(f"  ERROR: Model '{model_name}' not found in registry")
            return False

        versions = self.registry['models'][model_name]['versions']
        version_key = str(version)

        if version_key not in versions:
            print(f"  ERROR: Version {version} not found for {model_name}")
            return False

        old_stage = versions[version_key]['stage']
        versions[version_key]['stage'] = target_stage
        versions[version_key]['promoted_at'] = datetime.now().isoformat()
        self.registry['models'][model_name]['current_stage'] = target_stage

        promotion_record = {
            'model_name': model_name,
            'version': version,
            'from_stage': old_stage,
            'to_stage': target_stage,
            'promoted_at': datetime.now().isoformat(),
            'promoted_by': DOMINO_PROJECT_OWNER or 'system',
            'reason': reason,
        }
        self.registry['promotion_history'].append(promotion_record)
        self._save_registry()

        print(f"  Promoted: {model_name} v{version}: {old_stage} -> {target_stage}")
        if reason:
            print(f"  Reason: {reason}")

        # Also transition in MLflow Model Registry
        self._promote_in_mlflow(model_name, target_stage)

        return True

    def _promote_in_mlflow(self, model_name: str, target_stage: str):
        """Transition model stage in MLflow Registry (updates Models tab)."""
        if not self.mlflow_enabled:
            return

        stage_map = {
            'production': 'Production',
            'staging': 'Staging',
            'archived': 'Archived',
            'development': 'None',
        }
        mlflow_stage = stage_map.get(target_stage, 'None')

        try:
            model_versions = self.mlflow_client.get_latest_versions(model_name)
            if model_versions:
                latest_mv = model_versions[-1]
                self.mlflow_client.transition_model_version_stage(
                    name=model_name,
                    version=latest_mv.version,
                    stage=mlflow_stage
                )
                print(f"    [MLflow] Stage updated to '{mlflow_stage}' in Models tab")
        except Exception as e:
            print(f"    [MLflow] Stage transition skipped: {e}")

    def get_model(self, model_name: str, version: Optional[int] = None,
                  stage: Optional[str] = None) -> Optional[dict]:
        """Retrieve a model by name, version, or stage."""
        if model_name not in self.registry['models']:
            return None

        model_info = self.registry['models'][model_name]
        versions = model_info['versions']

        if version:
            return versions.get(str(version))
        elif stage:
            for v_info in versions.values():
                if v_info['stage'] == stage:
                    return v_info
        else:
            latest = model_info.get('latest_version', 1)
            return versions.get(str(latest))

    def list_models(self) -> list:
        """List all registered models."""
        models = []
        for name, info in self.registry['models'].items():
            models.append({
                'name': name,
                'latest_version': info.get('latest_version'),
                'stage': info.get('current_stage'),
                'versions_count': len(info['versions']),
            })
        return models

    def get_promotion_history(self, model_name: Optional[str] = None) -> list:
        """Get promotion history for a model or all models."""
        history = self.registry.get('promotion_history', [])
        if model_name:
            return [h for h in history if h['model_name'] == model_name]
        return history


def migrate_to_domino_registry():
    """Migrate all models from legacy registry to Domino."""
    print("\n" + "=" * 60)
    print("MIGRATING MODELS TO DOMINO REGISTRY")
    print("(Models will appear in Domino Models tab via MLflow)")
    print("=" * 60)

    registry = DominoModelRegistry()

    catalog_path = os.path.join(LEGACY_REGISTRY_DIR, 'registry_catalog.json')
    if not os.path.exists(catalog_path):
        print("ERROR: Legacy registry not found. Run legacy_registry_export.py first.")
        return

    with open(catalog_path, 'r') as f:
        catalog = json.load(f)

    print(f"\nSource: {catalog['registry_source']}")
    print(f"Models to migrate: {catalog['total_models']}")

    for model_name in catalog['models']:
        model_dir = os.path.join(LEGACY_REGISTRY_DIR, model_name)
        metadata_path = os.path.join(model_dir, 'metadata.json')
        model_path = os.path.join(model_dir, 'model.pkl')

        if not os.path.exists(metadata_path):
            print(f"\n  SKIP: {model_name} (no metadata found)")
            continue

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        print(f"\n  Migrating: {model_name}")
        artifacts = [os.path.join(model_dir, a) for a in metadata.get('artifacts', [])]
        registry.register_model(model_name, metadata, model_path, artifacts)

    print(f"\n{'=' * 60}")
    print(f"Migration complete. {catalog['total_models']} models registered.")
    print(f"Registry stored at: {REGISTRY_PATH}")
    if registry.mlflow_enabled:
        print(f"Models visible in: Domino Models tab (via MLflow Model Registry)")
    print(f"{'=' * 60}")

    return registry


if __name__ == '__main__':
    migrate_to_domino_registry()
