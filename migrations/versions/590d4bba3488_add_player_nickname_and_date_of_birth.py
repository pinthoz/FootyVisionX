"""add player nickname and date_of_birth

Revision ID: 590d4bba3488
Revises: d7ec30e6f086
Create Date: 2026-09-02 13:15:54.608871
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '590d4bba3488'
down_revision: Union[str, None] = 'd7ec30e6f086'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('players', sa.Column('nickname', sa.String(length=120), nullable=True))
    op.add_column('players', sa.Column('date_of_birth', sa.Date(), nullable=True))
    op.create_index(op.f('ix_players_nickname'), 'players', ['nickname'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_players_nickname'), table_name='players')
    op.drop_column('players', 'date_of_birth')
    op.drop_column('players', 'nickname')
