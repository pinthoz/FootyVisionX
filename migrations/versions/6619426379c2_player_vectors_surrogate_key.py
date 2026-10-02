"""player_vectors surrogate key

Autogenerate saw the new `id` column but not the primary-key move off `player_id`;
Alembic does not diff primary keys. Leaving it would keep the old constraint and make
every rebuild fail on the players who changed league mid-season and therefore hold one
profile per competition.

The table is a derived cache — it is rebuilt by `footyvision index` — so it is dropped
and recreated rather than migrated in place.

Revision ID: 6619426379c2
Revises: 2be893cc302e
Create Date: 2026-09-03 20:57:11.000000
"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "6619426379c2"
down_revision: Union[str, None] = "2be893cc302e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _create(with_surrogate_key: bool) -> None:
    columns = [
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("vector", sa.LargeBinary(), nullable=False),
        sa.Column("dim", sa.Integer(), nullable=False),
        sa.Column("embed_model", sa.String(length=200), nullable=False),
        sa.Column("foot", sa.String(length=10), nullable=True),
        sa.Column("age", sa.Float(), nullable=True),
        sa.Column("position_group", sa.String(length=20), nullable=True),
    ]
    if with_surrogate_key:
        key = [
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("player_id", sa.Integer(), nullable=False),
        ]
        constraints = [
            sa.PrimaryKeyConstraint("id"),
            sa.ForeignKeyConstraint(["player_id"], ["players.id"]),
        ]
    else:
        key = [sa.Column("player_id", sa.Integer(), nullable=False)]
        constraints = [
            sa.PrimaryKeyConstraint("player_id"),
            sa.ForeignKeyConstraint(["player_id"], ["players.id"]),
        ]

    op.create_table("player_vectors", *key, *columns, *constraints)
    op.create_index(
        op.f("ix_player_vectors_embed_model"), "player_vectors", ["embed_model"], unique=False
    )
    if with_surrogate_key:
        op.create_index(
            op.f("ix_player_vectors_player_id"), "player_vectors", ["player_id"], unique=False
        )


def upgrade() -> None:
    op.drop_table("player_vectors")
    _create(with_surrogate_key=True)


def downgrade() -> None:
    op.drop_table("player_vectors")
    _create(with_surrogate_key=False)
