"""
Domino Model Registry Integration.
Registers exported models into Domino's Model Registry with full metadata,
versioning, and promotion workflows.
"""
import os
import json
import pickle
import requests
from datetime import datetime
from typing import Optional

from config import (
    DOMINO_API_HOST, DOMINO_API_KEY, DOMINO_PROJECT_ID,
    DOMINO_PROJECT_OWNER, DOMINO_PROJECT_NAME,
    LEGACY_REGISTRY_DIR, REGISTRY_PATH, PROMOTION_STAGES
)


class DominoModelRegistry:
    """Manage models in Domino's Model Registry."""

    def __init__(self):
        self.api_host = DOMINO_API_HOST
        self.api_key = DOMINO_API_KEY
        self.project_id = DOMINO_PROJECT_ID
        self.headers = {
            'X-Domino-Api-Key': self.api_key,
            'Content-Type': 'application/json',
        }
        self.registry = self._load_registry()

    def _load_registry(self) -> dict:
        """Load or initialize the model registry."""
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
        """Persist registry state."""
        os.makedirs(REGISTRY_PATH, exist_ok=True)
        registry_file = os.path.join(REGISTRY_PATH, 'registry.json')
        with open(registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def register_model(self, model_name: str, metadata: dict,
                       model_path: str, artifacts: list = None) -> dict:
        """
        Register a model in the Domino Model Registry.

        Args:
            model_name: Unique model identifier
            metadata: Model metadata (params, metrics, lineage)
            model_path: Path to serialized model file
            artifacts: List of additional artifact paths
        """
        version = metadata.get('version', 1)
        model_key = f"{model_name}_v{version}"

        # Create model entry
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

        # Store in registry
        if model_name not in self.registry['models']:
            self.registry['models'][model_name] = {'versions': {}}
        self.registry['models'][model_name]['versions'][str(version)] = entry
        self.registry['models'][model_name]['latest_version'] = version
        self.registry['models'][model_name]['current_stage'] = entry['stage']

        # Copy model artifact to registry storage
        model_store = os.path.join(REGISTRY_PATH, model_name, f'v{version}')
        os.makedirs(model_store, exist_ok=True)

        if os.path.exists(model_path):
            import shutil
            shutil.copy2(model_path, os.path.join(model_store, 'model.pkl'))

        # Save metadata alongside model
        with open(os.path.join(model_store, 'metadata.json'), 'w') as f:
            json.dump(entry, f, indent=2)

        self._save_registry()

        print(f"  Registered: {model_name} v{version} [{entry['stage']}]")
        print(f"  Stored at:  {model_store}")

        # Try Domino API registration
        self._register_via_api(entry)

        return entry

    def _register_via_api(self, entry: dict):
        """Attempt to register model via Domino Model Registry API."""
        if not self.api_host or not self.api_key:
            return

        url = f"{self.api_host}/api/modelManager/v2/models"
        payload = {
            'projectId': self.project_id,
            'name': entry['model_name'],
            'description': entry['description'],
        }

        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            if response.status_code in (200, 201, 409):
                if response.status_code == 409:
                    print(f"    (Model already exists in Domino registry)")
                else:
                    print(f"    (Registered in Domino API)")
        except Exception as e:
            print(f"    (API registration skipped: {e})")

    def promote_model(self, model_name: str, version: int,
                      target_stage: str, reason: str = '') -> bool:
        """
        Promote a model to a new stage (dev → staging → production).

        Args:
            model_name: The model to promote
            version: Version number
            target_stage: Target stage (development/staging/production/archived)
            reason: Reason for promotion
        """
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

        # Log promotion history
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

        print(f"  Promoted: {model_name} v{version}: {old_stage} → {target_stage}")
        if reason:
            print(f"  Reason: {reason}")

        return True

    def get_model(self, model_name: str, version: Optional[int] = None,
                  stage: Optional[str] = None) -> Optional[dict]:
        """
        Retrieve a model by name, version, or stage.
        If no version/stage specified, returns the latest.
        """
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
    print("=" * 60)

    registry = DominoModelRegistry()

    # Read legacy registry catalog
    catalog_path = os.path.join(LEGACY_REGISTRY_DIR, 'registry_catalog.json')
    if not os.path.exists(catalog_path):
        print("ERROR: Legacy registry not found. Run legacy_registry_export.py first.")
        return

    with open(catalog_path, 'r') as f:
        catalog = json.load(f)

    print(f"\nSource: {catalog['registry_source']}")
    print(f"Models to migrate: {catalog['total_models']}")

    # Migrate each model
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
    print(f"{'=' * 60}")

    return registry


if __name__ == '__main__':
    migrate_to_domino_registry()
