"""initial memory audit tables

Revision ID: 0001_memory_audit_tables
Revises:
Create Date: 2026-07-01
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0001_memory_audit_tables"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "memory_records",
        sa.Column("memory_id", sa.String(length=128), primary_key=True),
        sa.Column("memory_kind", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("subject", sa.String(length=512), nullable=True),
        sa.Column("predicate", sa.String(length=512), nullable=True),
        sa.Column("object", sa.Text(), nullable=True),
        sa.Column("source", sa.String(length=256), nullable=True),
        sa.Column("object_json", sa.JSON(), nullable=False),
        sa.Column("meta_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("updated_at", sa.Float(), nullable=False),
    )
    op.create_index("ix_memory_records_memory_kind", "memory_records", ["memory_kind"])
    op.create_index("ix_memory_records_status", "memory_records", ["status"])
    op.create_index("ix_memory_records_subject", "memory_records", ["subject"])
    op.create_index("ix_memory_records_predicate", "memory_records", ["predicate"])
    op.create_index("ix_memory_records_source", "memory_records", ["source"])
    op.create_index("ix_memory_records_created_at", "memory_records", ["created_at"])
    op.create_index("ix_memory_records_updated_at", "memory_records", ["updated_at"])
    op.create_index("ix_memory_records_fact_key", "memory_records", ["subject", "predicate"])

    op.create_table(
        "memory_operations",
        sa.Column("id", sa.String(length=128), primary_key=True),
        sa.Column("operation", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("source", sa.String(length=256), nullable=True),
        sa.Column("item_id", sa.String(length=128), nullable=True),
        sa.Column("memory_id", sa.String(length=128), nullable=True),
        sa.Column("memory_kind", sa.String(length=64), nullable=True),
        sa.Column("before_json", sa.JSON(), nullable=True),
        sa.Column("after_json", sa.JSON(), nullable=True),
        sa.Column("meta_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("completed_at", sa.Float(), nullable=True),
    )
    op.create_index("ix_memory_operations_operation", "memory_operations", ["operation"])
    op.create_index("ix_memory_operations_status", "memory_operations", ["status"])
    op.create_index("ix_memory_operations_source", "memory_operations", ["source"])
    op.create_index("ix_memory_operations_item_id", "memory_operations", ["item_id"])
    op.create_index("ix_memory_operations_memory_id", "memory_operations", ["memory_id"])
    op.create_index("ix_memory_operations_memory_kind", "memory_operations", ["memory_kind"])
    op.create_index("ix_memory_operations_created_at", "memory_operations", ["created_at"])
    op.create_index("ix_memory_operations_completed_at", "memory_operations", ["completed_at"])

    op.create_table(
        "policy_decisions",
        sa.Column("id", sa.String(length=128), primary_key=True),
        sa.Column("operation_id", sa.String(length=128), nullable=True),
        sa.Column("stage", sa.String(length=64), nullable=False),
        sa.Column("policy", sa.String(length=128), nullable=False),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("accepted", sa.Boolean(), nullable=False),
        sa.Column("item_id", sa.String(length=128), nullable=True),
        sa.Column("memory_id", sa.String(length=128), nullable=True),
        sa.Column("memory_kind", sa.String(length=64), nullable=True),
        sa.Column("reason", sa.String(length=256), nullable=True),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.Column("meta_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
    )
    op.create_index("ix_policy_decisions_operation_id", "policy_decisions", ["operation_id"])
    op.create_index("ix_policy_decisions_stage", "policy_decisions", ["stage"])
    op.create_index("ix_policy_decisions_policy", "policy_decisions", ["policy"])
    op.create_index("ix_policy_decisions_action", "policy_decisions", ["action"])
    op.create_index("ix_policy_decisions_accepted", "policy_decisions", ["accepted"])
    op.create_index("ix_policy_decisions_item_id", "policy_decisions", ["item_id"])
    op.create_index("ix_policy_decisions_memory_id", "policy_decisions", ["memory_id"])
    op.create_index("ix_policy_decisions_memory_kind", "policy_decisions", ["memory_kind"])
    op.create_index("ix_policy_decisions_reason", "policy_decisions", ["reason"])
    op.create_index("ix_policy_decisions_created_at", "policy_decisions", ["created_at"])

    op.create_table(
        "review_candidates",
        sa.Column("id", sa.String(length=128), primary_key=True),
        sa.Column("operation_id", sa.String(length=128), nullable=True),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("item_id", sa.String(length=128), nullable=True),
        sa.Column("memory_id", sa.String(length=128), nullable=True),
        sa.Column("memory_kind", sa.String(length=64), nullable=True),
        sa.Column("source", sa.String(length=256), nullable=True),
        sa.Column("reason", sa.String(length=256), nullable=True),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.Column("memory_object_json", sa.JSON(), nullable=True),
        sa.Column("meta_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
    )
    op.create_index("ix_review_candidates_operation_id", "review_candidates", ["operation_id"])
    op.create_index("ix_review_candidates_status", "review_candidates", ["status"])
    op.create_index("ix_review_candidates_item_id", "review_candidates", ["item_id"])
    op.create_index("ix_review_candidates_memory_id", "review_candidates", ["memory_id"])
    op.create_index("ix_review_candidates_memory_kind", "review_candidates", ["memory_kind"])
    op.create_index("ix_review_candidates_source", "review_candidates", ["source"])
    op.create_index("ix_review_candidates_reason", "review_candidates", ["reason"])
    op.create_index("ix_review_candidates_created_at", "review_candidates", ["created_at"])


def downgrade() -> None:
    op.drop_table("review_candidates")
    op.drop_table("policy_decisions")
    op.drop_table("memory_operations")
    op.drop_table("memory_records")
