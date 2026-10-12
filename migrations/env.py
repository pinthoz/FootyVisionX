"""Alembic environment — wires migrations to the app's settings and metadata."""
from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from footyvision.config import get_settings
from footyvision.db.base import Base
import footyvision.db.models  # noqa: F401  (import to register all tables on Base.metadata)

config = context.config
# The URL passes through configparser, which reads "%" as the start of an interpolation.
# A URL-encoded password is full of them (RDS generates passwords with symbols), so each
# is doubled here and read back as a single "%".
config.set_main_option("sqlalchemy.url", get_settings().sqlalchemy_url.replace("%", "%%"))

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata, compare_type=True
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
