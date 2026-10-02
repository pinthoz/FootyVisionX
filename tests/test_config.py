from __future__ import annotations

from footyvision.config import Settings


def test_sqlalchemy_url_is_built_from_parts():
    s = Settings(
        postgres_user="u",
        postgres_password="p",
        postgres_host="h",
        postgres_port=1234,
        postgres_db="d",
        database_url=None,
    )
    assert s.sqlalchemy_url == "postgresql+psycopg2://u:p@h:1234/d"


def test_explicit_database_url_wins():
    s = Settings(database_url="postgresql+psycopg2://x/y")
    assert s.sqlalchemy_url == "postgresql+psycopg2://x/y"


def test_a_bare_postgres_url_is_pinned_to_the_installed_driver():
    """SQLAlchemy 2.1 made psycopg 3 the default for `postgresql://`, and the deploy that
    picked it up failed at import with "No module named 'psycopg'"."""
    from footyvision.config import Settings

    for given in ("postgresql://u:p@h/db", "postgres://u:p@h/db"):
        assert Settings(database_url=given).sqlalchemy_url == "postgresql+psycopg2://u:p@h/db"
    for kept in ("postgresql+psycopg2://u:p@h/db", "sqlite:///x.db"):
        assert Settings(database_url=kept).sqlalchemy_url == kept
