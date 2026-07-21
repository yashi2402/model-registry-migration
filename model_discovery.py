"""
Model Discoverability Across Teams.
Provides search, catalog, and sharing capabilities for models
registered in the Domino Model Registry.
"""
import os
import json
from datetime import datetime
from typing import Optional

from config import REGISTRY_PATH, DOMINO_PROJECT_OWNER
from domino_registry import DominoModelRegistry


class ModelDiscovery:
    """Enable model discoverability and sharing across teams."""

    def __init__(self):
        self.registry = DominoModelRegistry()
        self.catalog_path = os.path.join(REGISTRY_PATH, 'model_catalog.json')

    def generate_catalog(self) -> dict:
        """
        Generate a searchable model catalog for cross-team discovery.
        This catalog provides a centralized view of all available models.
        """
        models = self.registry.list_models()
        catalog_entries = []

        for model_info in models:
            model_name = model_info['name']
            latest = self.registry.get_model(model_name)
            if not latest:
                continue

            entry = {
                'model_name': model_name,
                'display_name': model_name.replace('-', ' ').title(),
                'description': latest.get('description', ''),
                'model_type': latest.get('model_type', ''),
                'framework': latest.get('framework', ''),
                'current_stage': latest.get('stage', 'unknown'),
                'latest_version': model_info['latest_version'],
                'total_versions': model_info['versions_count'],
                'owner': latest.get('created_by', 'unknown'),
                'use_case': latest.get('tags', [''])[0] if latest.get('tags') else '',
                'key_metrics': latest.get('metrics', {}),
                'training_data_source': latest.get('training_data', {}).get('source', ''),
                'last_updated': latest.get('lineage', {}).get('original_updated', ''),
                'migrated_from': 'GPS/Davnic MLflow',
                'searchable_tags': latest.get('tags', []) + [
                    latest.get('model_type', ''),
                    latest.get('stage', ''),
                    latest.get('created_by', ''),
                ],
            }
            catalog_entries.append(entry)

        catalog = {
            'catalog_version': '1.0',
            'generated_at': datetime.now().isoformat(),
            'generated_by': DOMINO_PROJECT_OWNER or 'system',
            'total_models': len(catalog_entries),
            'models': catalog_entries,
            'stages_summary': self._get_stage_summary(catalog_entries),
            'teams_summary': self._get_team_summary(catalog_entries),
        }

        with open(self.catalog_path, 'w') as f:
            json.dump(catalog, f, indent=2)

        return catalog

    def _get_stage_summary(self, entries: list) -> dict:
        """Summarize models by stage."""
        summary = {}
        for entry in entries:
            stage = entry['current_stage']
            if stage not in summary:
                summary[stage] = []
            summary[stage].append(entry['model_name'])
        return summary

    def _get_team_summary(self, entries: list) -> dict:
        """Summarize models by team/owner."""
        summary = {}
        for entry in entries:
            owner = entry['owner']
            if owner not in summary:
                summary[owner] = []
            summary[owner].append(entry['model_name'])
        return summary

    def search_models(self, query: str = '', stage: str = '',
                      framework: str = '', use_case: str = '') -> list:
        """
        Search models in the catalog.

        Args:
            query: Free-text search across name, description, tags
            stage: Filter by promotion stage
            framework: Filter by ML framework
            use_case: Filter by use case
        """
        if not os.path.exists(self.catalog_path):
            self.generate_catalog()

        with open(self.catalog_path, 'r') as f:
            catalog = json.load(f)

        results = catalog['models']

        if query:
            query_lower = query.lower()
            results = [
                m for m in results
                if query_lower in m['model_name'].lower()
                or query_lower in m['description'].lower()
                or any(query_lower in t.lower() for t in m.get('searchable_tags', []))
            ]

        if stage:
            results = [m for m in results if m['current_stage'] == stage]

        if framework:
            results = [m for m in results if framework.lower() in m['framework'].lower()]

        if use_case:
            results = [m for m in results if use_case.lower() in m.get('use_case', '').lower()]

        return results

    def get_model_card(self, model_name: str) -> dict:
        """
        Generate a model card — a standardized summary for model documentation.
        Used for governance, compliance, and team communication.
        """
        model = self.registry.get_model(model_name)
        if not model:
            return {}

        history = self.registry.get_promotion_history(model_name)

        card = {
            'model_card_version': '1.0',
            'generated_at': datetime.now().isoformat(),
            'model_details': {
                'name': model_name,
                'version': model.get('version'),
                'type': model.get('model_type'),
                'framework': model.get('framework'),
                'description': model.get('description'),
                'owner': model.get('created_by'),
                'stage': model.get('stage'),
            },
            'intended_use': {
                'primary_use': model.get('description', ''),
                'use_case': model.get('tags', [''])[0] if model.get('tags') else '',
                'users': 'Data science and ML engineering teams',
                'limitations': 'Trained on historical data; may not generalize to new patterns',
            },
            'training_data': model.get('training_data', {}),
            'performance_metrics': model.get('metrics', {}),
            'hyperparameters': model.get('parameters', {}),
            'lineage': model.get('lineage', {}),
            'promotion_history': history,
            'ethical_considerations': {
                'bias_evaluation': 'Pending — requires fairness audit',
                'data_privacy': 'No PII in training features',
            },
        }

        # Save model card
        cards_dir = os.path.join(REGISTRY_PATH, 'model_cards')
        os.makedirs(cards_dir, exist_ok=True)
        card_path = os.path.join(cards_dir, f'{model_name}_card.json')
        with open(card_path, 'w') as f:
            json.dump(card, f, indent=2)

        return card


def demo_discovery():
    """Demonstrate model discoverability features."""
    print("\n" + "=" * 60)
    print("MODEL DISCOVERABILITY & CROSS-TEAM SHARING")
    print("=" * 60)

    discovery = ModelDiscovery()

    # Generate catalog
    print("\n[1] Generating model catalog...")
    catalog = discovery.generate_catalog()
    print(f"    Models cataloged: {catalog['total_models']}")

    # Show catalog
    print("\n[2] Model Catalog:")
    print(f"    {'Name':<30} {'Stage':<15} {'Type':<30} {'Owner'}")
    print(f"    {'-'*30} {'-'*15} {'-'*30} {'-'*20}")
    for m in catalog['models']:
        print(f"    {m['model_name']:<30} {m['current_stage']:<15} "
              f"{m['model_type']:<30} {m['owner']}")

    # Search
    print("\n[3] Search: 'fraud'")
    results = discovery.search_models(query='fraud')
    for r in results:
        print(f"    Found: {r['model_name']} ({r['current_stage']})")

    print("\n    Search: stage='production'")
    results = discovery.search_models(stage='production')
    for r in results:
        print(f"    Found: {r['model_name']} (v{r['latest_version']})")

    # Model cards
    print("\n[4] Generating model cards...")
    for model_info in discovery.registry.list_models():
        card = discovery.get_model_card(model_info['name'])
        print(f"    Card: {model_info['name']} — {card['model_details']['description'][:50]}...")

    # Stage summary
    print("\n[5] Models by Stage:")
    for stage, models in catalog['stages_summary'].items():
        print(f"    {stage}: {', '.join(models)}")

    # Team summary
    print("\n[6] Models by Team:")
    for team, models in catalog['teams_summary'].items():
        print(f"    {team}: {', '.join(models)}")

    print(f"\n{'=' * 60}")
    print("Catalog and model cards available at:")
    print(f"  {REGISTRY_PATH}/model_catalog.json")
    print(f"  {REGISTRY_PATH}/model_cards/")
    print(f"{'=' * 60}")

    return discovery


if __name__ == '__main__':
    demo_discovery()
