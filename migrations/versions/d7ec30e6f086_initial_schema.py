"""Initial schema.

Creates the seven core tables: competitions, seasons, teams, players, matches,
player_match_stats and player_season_stats. StatsBomb IDs are reused as primary keys so
the ETL can upsert by natural key.

Note: `footyvision init-db` still uses `create_all` for a fast first run; this revision
exists so schema changes from here on are versioned.

Revision ID: d7ec30e6f086
Revises:
Create Date: 2026-09-01
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'd7ec30e6f086'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('competitions',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=120), nullable=False),
    sa.Column('country', sa.String(length=120), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('players',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=120), nullable=False),
    sa.Column('country', sa.String(length=120), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_players_name'), 'players', ['name'], unique=False)
    op.create_table('teams',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=120), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('matches',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('competition_id', sa.Integer(), nullable=False),
    sa.Column('sb_season_id', sa.Integer(), nullable=False),
    sa.Column('match_date', sa.Date(), nullable=True),
    sa.Column('home_team_id', sa.Integer(), nullable=True),
    sa.Column('away_team_id', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['away_team_id'], ['teams.id'], ),
    sa.ForeignKeyConstraint(['competition_id'], ['competitions.id'], ),
    sa.ForeignKeyConstraint(['home_team_id'], ['teams.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_matches_competition_id'), 'matches', ['competition_id'], unique=False)
    op.create_index(op.f('ix_matches_sb_season_id'), 'matches', ['sb_season_id'], unique=False)
    op.create_table('player_season_stats',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('player_id', sa.Integer(), nullable=False),
    sa.Column('competition_id', sa.Integer(), nullable=False),
    sa.Column('sb_season_id', sa.Integer(), nullable=False),
    sa.Column('primary_position', sa.String(length=40), nullable=True),
    sa.Column('matches_played', sa.Integer(), nullable=False),
    sa.Column('minutes', sa.Float(), nullable=False),
    sa.Column('goals', sa.Float(), nullable=False),
    sa.Column('assists', sa.Float(), nullable=False),
    sa.Column('shots', sa.Float(), nullable=False),
    sa.Column('xg', sa.Float(), nullable=False),
    sa.Column('passes', sa.Float(), nullable=False),
    sa.Column('passes_completed', sa.Float(), nullable=False),
    sa.Column('progressive_passes', sa.Float(), nullable=False),
    sa.Column('dribbles', sa.Float(), nullable=False),
    sa.Column('dribbles_completed', sa.Float(), nullable=False),
    sa.Column('carries', sa.Float(), nullable=False),
    sa.Column('progressive_carries', sa.Float(), nullable=False),
    sa.Column('tackles', sa.Float(), nullable=False),
    sa.Column('interceptions', sa.Float(), nullable=False),
    sa.Column('blocks', sa.Float(), nullable=False),
    sa.Column('clearances', sa.Float(), nullable=False),
    sa.Column('ball_recoveries', sa.Float(), nullable=False),
    sa.Column('pressures', sa.Float(), nullable=False),
    sa.Column('goals_per90', sa.Float(), nullable=False),
    sa.Column('assists_per90', sa.Float(), nullable=False),
    sa.Column('shots_per90', sa.Float(), nullable=False),
    sa.Column('xg_per90', sa.Float(), nullable=False),
    sa.Column('passes_per90', sa.Float(), nullable=False),
    sa.Column('passes_completed_per90', sa.Float(), nullable=False),
    sa.Column('progressive_passes_per90', sa.Float(), nullable=False),
    sa.Column('dribbles_per90', sa.Float(), nullable=False),
    sa.Column('dribbles_completed_per90', sa.Float(), nullable=False),
    sa.Column('carries_per90', sa.Float(), nullable=False),
    sa.Column('progressive_carries_per90', sa.Float(), nullable=False),
    sa.Column('tackles_per90', sa.Float(), nullable=False),
    sa.Column('interceptions_per90', sa.Float(), nullable=False),
    sa.Column('blocks_per90', sa.Float(), nullable=False),
    sa.Column('clearances_per90', sa.Float(), nullable=False),
    sa.Column('ball_recoveries_per90', sa.Float(), nullable=False),
    sa.Column('pressures_per90', sa.Float(), nullable=False),
    sa.ForeignKeyConstraint(['competition_id'], ['competitions.id'], ),
    sa.ForeignKeyConstraint(['player_id'], ['players.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('player_id', 'competition_id', 'sb_season_id', name='uq_player_season')
    )
    op.create_index(op.f('ix_player_season_stats_competition_id'), 'player_season_stats', ['competition_id'], unique=False)
    op.create_index(op.f('ix_player_season_stats_player_id'), 'player_season_stats', ['player_id'], unique=False)
    op.create_index(op.f('ix_player_season_stats_sb_season_id'), 'player_season_stats', ['sb_season_id'], unique=False)
    op.create_table('seasons',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('competition_id', sa.Integer(), nullable=False),
    sa.Column('sb_season_id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=60), nullable=False),
    sa.ForeignKeyConstraint(['competition_id'], ['competitions.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('competition_id', 'sb_season_id', name='uq_comp_season')
    )
    op.create_index(op.f('ix_seasons_competition_id'), 'seasons', ['competition_id'], unique=False)
    op.create_table('player_match_stats',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('match_id', sa.Integer(), nullable=False),
    sa.Column('player_id', sa.Integer(), nullable=False),
    sa.Column('team_id', sa.Integer(), nullable=False),
    sa.Column('position', sa.String(length=40), nullable=True),
    sa.Column('minutes', sa.Float(), nullable=False),
    sa.Column('goals', sa.Float(), nullable=False),
    sa.Column('assists', sa.Float(), nullable=False),
    sa.Column('shots', sa.Float(), nullable=False),
    sa.Column('xg', sa.Float(), nullable=False),
    sa.Column('passes', sa.Float(), nullable=False),
    sa.Column('passes_completed', sa.Float(), nullable=False),
    sa.Column('progressive_passes', sa.Float(), nullable=False),
    sa.Column('dribbles', sa.Float(), nullable=False),
    sa.Column('dribbles_completed', sa.Float(), nullable=False),
    sa.Column('carries', sa.Float(), nullable=False),
    sa.Column('progressive_carries', sa.Float(), nullable=False),
    sa.Column('tackles', sa.Float(), nullable=False),
    sa.Column('interceptions', sa.Float(), nullable=False),
    sa.Column('blocks', sa.Float(), nullable=False),
    sa.Column('clearances', sa.Float(), nullable=False),
    sa.Column('ball_recoveries', sa.Float(), nullable=False),
    sa.Column('pressures', sa.Float(), nullable=False),
    sa.ForeignKeyConstraint(['match_id'], ['matches.id'], ),
    sa.ForeignKeyConstraint(['player_id'], ['players.id'], ),
    sa.ForeignKeyConstraint(['team_id'], ['teams.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('match_id', 'player_id', name='uq_match_player')
    )
    op.create_index(op.f('ix_player_match_stats_match_id'), 'player_match_stats', ['match_id'], unique=False)
    op.create_index(op.f('ix_player_match_stats_player_id'), 'player_match_stats', ['player_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_player_match_stats_player_id'), table_name='player_match_stats')
    op.drop_index(op.f('ix_player_match_stats_match_id'), table_name='player_match_stats')
    op.drop_table('player_match_stats')
    op.drop_index(op.f('ix_seasons_competition_id'), table_name='seasons')
    op.drop_table('seasons')
    op.drop_index(op.f('ix_player_season_stats_sb_season_id'), table_name='player_season_stats')
    op.drop_index(op.f('ix_player_season_stats_player_id'), table_name='player_season_stats')
    op.drop_index(op.f('ix_player_season_stats_competition_id'), table_name='player_season_stats')
    op.drop_table('player_season_stats')
    op.drop_index(op.f('ix_matches_sb_season_id'), table_name='matches')
    op.drop_index(op.f('ix_matches_competition_id'), table_name='matches')
    op.drop_table('matches')
    op.drop_table('teams')
    op.drop_index(op.f('ix_players_name'), table_name='players')
    op.drop_table('players')
    op.drop_table('competitions')
