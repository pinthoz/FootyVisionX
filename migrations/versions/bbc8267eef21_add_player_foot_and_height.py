"""add player foot and height

Revision ID: bbc8267eef21
Revises: 590d4bba3488
Create Date: 2026-09-02 22:41:02.115330
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'bbc8267eef21'
down_revision: Union[str, None] = '590d4bba3488'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('players', sa.Column('foot', sa.String(length=10), nullable=True))
    op.add_column('players', sa.Column('height_cm', sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column('players', 'height_cm')
    op.drop_column('players', 'foot')
