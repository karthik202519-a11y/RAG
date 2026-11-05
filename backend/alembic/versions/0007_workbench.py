"""Versioned pipeline recipes and durable executions."""
from alembic import op
import sqlalchemy as sa
revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("pipeline_recipes", sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("project_id", sa.String(32), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(160), nullable=False), sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("project_id", "name", "version"))
    op.create_index("ix_pipeline_recipes_project_id", "pipeline_recipes", ["project_id"])
    op.create_table("workbench_runs", sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("project_id", sa.String(32), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("kind", sa.String(20), nullable=False), sa.Column("name", sa.String(160), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False), sa.Column("status", sa.String(20), nullable=False),
        sa.Column("artifacts", sa.JSON(), nullable=False), sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False))
    op.create_index("ix_workbench_runs_project_id", "workbench_runs", ["project_id"])


def downgrade():
    op.drop_table("workbench_runs")
    op.drop_table("pipeline_recipes")
