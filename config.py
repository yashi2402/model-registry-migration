"""
Configuration for Model Registry Migration.
Central config for all scripts in this project.
"""
import os

# Domino environment
DOMINO_API_HOST = os.environ.get('DOMINO_API_PROXY', os.environ.get('DOMINO_API_HOST', ''))
DOMINO_API_KEY = os.environ.get('DOMINO_USER_API_KEY', '')
DOMINO_PROJECT_ID = os.environ.get('DOMINO_PROJECT_ID', '')
DOMINO_PROJECT_OWNER = os.environ.get('DOMINO_PROJECT_OWNER', '')
DOMINO_PROJECT_NAME = os.environ.get('DOMINO_PROJECT_NAME', '')
DOMINO_RUN_ID = os.environ.get('DOMINO_RUN_ID', '')

# Paths
LEGACY_REGISTRY_DIR = os.path.join(os.path.dirname(__file__), 'legacy_registry')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')
EXPERIMENTS_DIR = '/mnt/data/model-registry/experiments'
REGISTRY_PATH = '/mnt/data/model-registry/registry'

# Model training config
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Model promotion stages
PROMOTION_STAGES = ['development', 'staging', 'production', 'archived']
