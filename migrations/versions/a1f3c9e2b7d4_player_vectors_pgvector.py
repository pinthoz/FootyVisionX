"""player_vectors.embedding as a pgvector column with an HNSW index

Revision ID: a1f3c9e2b7d4
Revises: 40e7367870b9
"""

from typing import Sequence, Union

import numpy as np
import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "a1f3c9e2b7d4"
down_revision: Union[str, None] = "40e7367870b9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.add_column("player_vectors", sa.Column("embedding", Vector(768), nullable=True))

    # Copy every stored vector across: bytes -> float32 -> pgvector's text form.
    conn = op.get_bind()
    rows = conn.execute(sa.text("SELECT id, vector FROM player_vectors")).all()
    for row_id, blob in rows:
        vector = np.frombuffer(blob, dtype=np.float32).tolist()
        conn.execute(
            sa.text("UPDATE player_vectors SET embedding = CAST(:v AS vector) WHERE id = :id"),
            {"v": str(vector), "id": row_id},
        )

    op.execute(
        "CREATE INDEX player_vectors_embedding_hnsw ON player_vectors "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.drop_index("player_vectors_embedding_hnsw", table_name="player_vectors")
    op.drop_column("player_vectors", "embedding")
