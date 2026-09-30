"""Add reassessment trigger reason to versioned plans.

Revision ID: 8f8b1f3c3f1a
Revises: c5324edbeac4
"""

from alembic import op
import sqlalchemy as sa

revision = "8f8b1f3c3f1a"
down_revision = "c5324edbeac4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("recovery_plan", sa.Column("replan_reason", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("recovery_plan", "replan_reason")
