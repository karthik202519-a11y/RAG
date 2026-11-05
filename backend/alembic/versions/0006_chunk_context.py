"""contextual embedding sentence on chunks

Revision ID: 0006
Revises: 0005
"""
from alembic import op
import sqlalchemy as sa

revision = '0006'
down_revision = '0005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('index_chunks', sa.Column('context', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('index_chunks', 'context')
