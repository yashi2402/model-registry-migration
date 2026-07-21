"""
Model Promotion Workflow.
Implements the dev -> staging -> production promotion pipeline
with validation gates and approval tracking.
Promotions are reflected in Domino's Models tab via MLflow stage transitions.
"""
import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional

from config import REGISTRY_PATH, PROMOTION_STAGES
from domino_registry import DominoModelRegistry

APPROVER_EMAIL = 'yashi_rahangdale@epam.com'


def send_notification_email(subject: str, body: str, to_email: str = APPROVER_EMAIL):
    """Send email notification about promotion decision."""
    try:
        msg = MIMEMultipart()
        msg['From'] = 'model-registry@domino.tech'
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Try to send via localhost SMTP (works in many enterprise environments)
        with smtplib.SMTP('localhost', 25, timeout=5) as server:
            server.sendmail(msg['From'], to_email, msg.as_string())
        print(f"  [EMAIL] Notification sent to {to_email}")
    except Exception as e:
        print(f"  [EMAIL] Notification skipped (SMTP not configured: {type(e).__name__})")
        print(f"  [EMAIL] Would have sent to: {to_email}")
        print(f"  [EMAIL] Subject: {subject}")


class ModelPromotionManager:
    """Manages model promotion workflows with validation gates."""

    def __init__(self):
        self.registry = DominoModelRegistry()
        self.promotion_rules = {
            'development_to_staging': {
                'min_accuracy': 0.80,
                'min_f1': 0.75,
                'requires_approval': False,
            },
            'staging_to_production': {
                'min_accuracy': 0.85,
                'min_f1': 0.80,
                'min_roc_auc': 0.85,
                'requires_approval': True,
                'approvers': ['yashi_rahangdale@epam.com'],
            },
            'production_to_archived': {
                'requires_approval': True,
                'approvers': ['yashi_rahangdale@epam.com'],
            },
        }

    def validate_promotion(self, model_name: str, version: int,
                           target_stage: str) -> dict:
        """
        Validate whether a model meets promotion criteria.
        Returns validation result with pass/fail and details.
        """
        model = self.registry.get_model(model_name, version=version)
        if not model:
            return {'valid': False, 'reason': 'Model not found'}

        current_stage = model['stage']
        transition_key = f"{current_stage}_to_{target_stage}"

        if transition_key not in self.promotion_rules:
            current_idx = PROMOTION_STAGES.index(current_stage) if current_stage in PROMOTION_STAGES else -1
            target_idx = PROMOTION_STAGES.index(target_stage) if target_stage in PROMOTION_STAGES else -1

            if target_idx <= current_idx and target_stage != 'archived':
                return {'valid': False, 'reason': f'Cannot demote from {current_stage} to {target_stage}'}
            return {'valid': True, 'checks': [], 'warnings': ['No validation rules for this transition']}

        rules = self.promotion_rules[transition_key]
        metrics = model.get('metrics', {})
        checks = []
        passed = True

        if 'min_accuracy' in rules:
            acc = metrics.get('accuracy', 0)
            check = {
                'metric': 'accuracy',
                'required': rules['min_accuracy'],
                'actual': acc,
                'passed': acc >= rules['min_accuracy'],
            }
            checks.append(check)
            if not check['passed']:
                passed = False

        if 'min_f1' in rules:
            f1 = metrics.get('f1_score', metrics.get('cv_mean_accuracy', 0))
            check = {
                'metric': 'f1_score',
                'required': rules['min_f1'],
                'actual': f1,
                'passed': f1 >= rules['min_f1'],
            }
            checks.append(check)
            if not check['passed']:
                passed = False

        if 'min_roc_auc' in rules:
            auc = metrics.get('roc_auc', 0)
            check = {
                'metric': 'roc_auc',
                'required': rules['min_roc_auc'],
                'actual': auc,
                'passed': auc >= rules['min_roc_auc'],
            }
            checks.append(check)
            if not check['passed']:
                passed = False

        return {
            'valid': passed,
            'model_name': model_name,
            'version': version,
            'from_stage': current_stage,
            'to_stage': target_stage,
            'checks': checks,
            'requires_approval': rules.get('requires_approval', False),
            'approvers': rules.get('approvers', []),
            'validated_at': datetime.now().isoformat(),
        }

    def promote(self, model_name: str, version: int,
                target_stage: str, reason: str = '',
                approved_by: str = 'auto') -> dict:
        """
        Execute model promotion with validation and approval.
        If approval is required, asks the approver interactively.
        Updates both local registry and MLflow Model Registry (Domino Models tab).
        """
        validation = self.validate_promotion(model_name, version, target_stage)

        if not validation['valid']:
            print(f"  BLOCKED: {model_name} v{version} -> {target_stage}")
            print(f"  Reason: Failed validation checks")
            for check in validation.get('checks', []):
                status = "PASS" if check['passed'] else "FAIL"
                print(f"    [{status}] {check['metric']}: {check['actual']} (min: {check['required']})")
            return {'promoted': False, 'validation': validation}

        # If approval is required, ask for it
        if validation.get('requires_approval', False):
            approvers = validation.get('approvers', [])
            print(f"  APPROVAL REQUIRED from: {', '.join(approvers)}")
            print(f"  Model: {model_name} v{version}")
            print(f"  Transition: {validation['from_stage']} -> {target_stage}")
            print(f"  Reason: {reason}")
            for check in validation.get('checks', []):
                status = "PASS" if check['passed'] else "FAIL"
                print(f"    [{status}] {check['metric']}: {check['actual']} (min: {check['required']})")

            approval = input(f"\n  [{approvers[0]}] Do you approve this promotion? (yes/no): ").strip().lower()

            if approval not in ('yes', 'y'):
                print(f"  REJECTED: Promotion denied by approver")
                send_notification_email(
                    subject=f"[REJECTED] Model Promotion: {model_name} v{version} -> {target_stage}",
                    body=f"Model promotion was REJECTED.\n\n"
                         f"Model: {model_name}\n"
                         f"Version: {version}\n"
                         f"Requested transition: {validation['from_stage']} -> {target_stage}\n"
                         f"Reason: {reason}\n"
                         f"Rejected by: {approvers[0]}\n"
                         f"Time: {datetime.now().isoformat()}\n"
                )
                record = {
                    'model_name': model_name,
                    'version': version,
                    'target_stage': target_stage,
                    'validation': validation,
                    'promoted': False,
                    'reason': reason,
                    'rejected_by': approvers[0],
                    'timestamp': datetime.now().isoformat(),
                }
                records_dir = os.path.join(REGISTRY_PATH, 'promotion_records')
                os.makedirs(records_dir, exist_ok=True)
                record_file = os.path.join(
                    records_dir,
                    f"{model_name}_v{version}_{target_stage}_REJECTED_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                with open(record_file, 'w') as f:
                    json.dump(record, f, indent=2)
                return {'promoted': False, 'validation': validation, 'rejected_by': approvers[0]}

            approved_by = approvers[0]
            print(f"  APPROVED by: {approved_by}")
            send_notification_email(
                subject=f"[APPROVED] Model Promotion: {model_name} v{version} -> {target_stage}",
                body=f"Model promotion was APPROVED.\n\n"
                     f"Model: {model_name}\n"
                     f"Version: {version}\n"
                     f"Transition: {validation['from_stage']} -> {target_stage}\n"
                     f"Reason: {reason}\n"
                     f"Approved by: {approved_by}\n"
                     f"Time: {datetime.now().isoformat()}\n"
            )

        success = self.registry.promote_model(
            model_name, version, target_stage, reason
        )

        record = {
            'model_name': model_name,
            'version': version,
            'target_stage': target_stage,
            'validation': validation,
            'promoted': success,
            'reason': reason,
            'approved_by': approved_by,
            'timestamp': datetime.now().isoformat(),
        }

        records_dir = os.path.join(REGISTRY_PATH, 'promotion_records')
        os.makedirs(records_dir, exist_ok=True)
        record_file = os.path.join(
            records_dir,
            f"{model_name}_v{version}_{target_stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(record_file, 'w') as f:
            json.dump(record, f, indent=2)

        return {'promoted': success, 'validation': validation, 'record': record_file}


def run_promotion_workflow():
    """Demonstrate the full promotion workflow."""
    print("\n" + "=" * 60)
    print("MODEL PROMOTION WORKFLOW (dev -> staging -> production)")
    print("(Stage changes reflected in Domino Models tab)")
    print("=" * 60)

    manager = ModelPromotionManager()

    models = manager.registry.list_models()
    print(f"\nRegistered models: {len(models)}")
    for m in models:
        print(f"  * {m['name']} (v{m['latest_version']}, stage: {m['stage']})")

    # Promote churn model from staging -> production
    print("\n--- Promotion 1: customer-churn-model (staging -> production) ---")
    result = manager.promote(
        'customer-churn-model', 3, 'production',
        reason='Passed A/B testing with 12% improvement over baseline',
        approved_by='yashi_rahangdale@epam.com'
    )
    for check in result['validation'].get('checks', []):
        status = "PASS" if check['passed'] else "FAIL"
        print(f"    [{status}] {check['metric']}: {check['actual']} (min: {check['required']})")

    # Promote transaction-classifier from development -> staging
    print("\n--- Promotion 2: transaction-classifier (development -> staging) ---")
    result = manager.promote(
        'transaction-classifier', 1, 'staging',
        reason='Feature complete, ready for integration testing'
    )
    for check in result['validation'].get('checks', []):
        status = "PASS" if check['passed'] else "FAIL"
        print(f"    [{status}] {check['metric']}: {check['actual']} (min: {check['required']})")

    # Show promotion history
    print("\n--- Promotion History ---")
    history = manager.registry.get_promotion_history()
    for h in history:
        print(f"  {h['model_name']} v{h['version']}: "
              f"{h['from_stage']} -> {h['to_stage']} "
              f"(by {h['promoted_by']}, {h['promoted_at'][:10]})")

    print(f"\n{'=' * 60}")
    return manager


if __name__ == '__main__':
    run_promotion_workflow()
