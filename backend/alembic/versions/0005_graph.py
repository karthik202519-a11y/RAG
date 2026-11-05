"""knowledge graph tables

Revision ID: 0005
Revises: 0004
"""
from alembic import op
import sqlalchemy as sa

revision = '0005'
down_revision = '0004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('graph_builds',
        sa.Column('index_id', sa.String(length=32), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('model', sa.String(length=100), nullable=False),
        sa.Column('chunks_requested', sa.Integer(), nullable=False),
        sa.Column('chunks_processed', sa.Integer(), nullable=False),
        sa.Column('chunks_failed', sa.Integer(), nullable=False),
        sa.Column('relations_dropped', sa.Integer(), nullable=False),
        sa.Column('tokens', sa.Integer(), nullable=False),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('seconds', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['index_id'], ['indexes.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('index_id'))
    op.create_table('graph_entities',
        sa.Column('id', sa.String(length=32), nullable=False),
        sa.Column('index_id', sa.String(length=32), nullable=False),
        sa.Column('name', sa.String(length=300), nullable=False),
        sa.Column('norm', sa.String(length=300), nullable=False),
        sa.Column('etype', sa.String(length=60), nullable=False),
        sa.Column('mentions', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['index_id'], ['indexes.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'))
    op.create_index('ix_graph_entities_index_id', 'graph_entities', ['index_id'])
    op.create_index('ix_graph_entities_norm', 'graph_entities', ['norm'])
    op.create_table('graph_edges',
        sa.Column('id', sa.String(length=32), nullable=False),
        sa.Column('index_id', sa.String(length=32), nullable=False),
        sa.Column('src_id', sa.String(length=32), nullable=False),
        sa.Column('dst_id', sa.String(length=32), nullable=False),
        sa.Column('relation', sa.String(length=200), nullable=False),
        sa.Column('chunk_id', sa.String(length=32), nullable=False),
        sa.ForeignKeyConstraint(['index_id'], ['indexes.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['src_id'], ['graph_entities.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['dst_id'], ['graph_entities.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'))
    for c in ('index_id', 'src_id', 'dst_id', 'chunk_id'):
        op.create_index(f'ix_graph_edges_{c}', 'graph_edges', [c])


def downgrade() -> None:
    op.drop_table('graph_edges')
    op.drop_table('graph_entities')
    op.drop_table('graph_builds')
