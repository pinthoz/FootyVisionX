"""The question bank the RAGAS evaluation runs against.

Fifty-eight hand-written questions was a decent probe and a thin sample: three or four per
category, where one odd score moves a category mean by ten points. The rest are built from
templates crossed with vocabularies taken from the database itself — both the only
practical way to reach two hundred and the more defensible one, since every nationality,
position and player name below was checked against the pool before it was written down.

Deterministic: a fixed seed and a fixed template list, so the set is identical on every
machine and a score can be compared against the last run rather than against a new sample.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    text: str
    kind: str


# Answerable in full, or answerable only with a caveat — both are graded as answers. Only
# the questions with nothing behind them at all are excluded from answer relevancy, since
# RAGAS scores a refusal as irrelevant by design.
REFUSAL_KINDS = ("unanswerable-metric", "unanswerable-player")


def is_refusal(kind: str) -> bool:
    return kind in REFUSAL_KINDS


# --- vocabularies, all verified against the pool -------------------------------------

# Verified by pinning each one through `store.mentioned` and keeping only the names that
# resolve to exactly one player. The candidates that did *not* survive are the lesson:
# "Bale" pins Balenziaga, "Aguero" pins Balaguero, "Suarez" pins eight different people.
PINNABLE: tuple[str, ...] = (
    "Messi",
    "Ronaldo",
    "Neymar",
    "Iniesta",
    "Griezmann",
    "Casimiro",
    "Putellas",
    "Kroos",
    "Bakambu",
    "Leno",
    "Modric",
    "Hazard",
    "Benzema",
    "Bonucci",
    "Buffon",
    "Higuain",
    "Perisic",
    "Vardy",
    "Payet",
    "Lacazette",
    "Bellarabi",
    "Alessandrini",
    "Cavani",
    "Verratti",
    "Marchisio",
    "Pjanic",
    "Oyarzabal",
    "Depay",
    "Alli",
    "Ozil",
    "Firmino",
    "Coutinho",
    "Mahrez",
    "Wendell",
    "Kampl",
)

# Nationalities with fifteen or more players, so a question about one has a real pool.
NATIONS_EN: tuple[str, ...] = (
    "Spanish",
    "Italian",
    "French",
    "English",
    "German",
    "Brazilian",
    "Argentine",
    "Swedish",
    "Senegalese",
    "Dutch",
    "Portuguese",
    "Serbian",
    "Belgian",
    "Scottish",
    "Danish",
    "Welsh",
    "Swiss",
    "Norwegian",
    "Austrian",
    "Irish",
    "Japanese",
    "Moroccan",
    "Polish",
    "Uruguayan",
    "Colombian",
    "Croatian",
    "Canadian",
    "Nigerian",
    "Chilean",
)
NATIONS_PT: tuple[str, ...] = (
    "espanhois",
    "italianos",
    "franceses",
    "ingleses",
    "alemaes",
    "brasileiros",
    "argentinos",
    "suecos",
    "senegaleses",
    "holandeses",
    "portugueses",
    "servios",
    "belgas",
    "escoceses",
    "dinamarqueses",
    "noruegueses",
    "austriacos",
    "irlandeses",
    "japoneses",
    "marroquinos",
    "polacos",
    "uruguaios",
    "colombianos",
    "croatas",
    "canadianos",
    "nigerianos",
    "chilenos",
)

ROLES_EN: tuple[str, ...] = (
    "wingers",
    "centre backs",
    "full-backs",
    "wing-backs",
    "goalkeepers",
    "defensive midfielders",
    "attacking midfielders",
    "centre forwards",
    "strikers",
)
ROLES_PT: tuple[str, ...] = (
    "extremos",
    "defesas centrais",
    "laterais",
    "alas",
    "guarda-redes",
    "medios defensivos",
    "medios ofensivos",
    "pontas de lanca",
    "avancados",
)

METRICS_EN: tuple[str, ...] = (
    "goals",
    "assists",
    "shots",
    "progressive passes",
    "dribbles",
    "carries",
    "tackles",
    "interceptions",
    "blocks",
    "clearances",
    "ball recoveries",
    "pressures",
)
METRICS_PT: tuple[str, ...] = (
    "golos",
    "assistencias",
    "remates",
    "passes progressivos",
    "dribles",
    "conducoes",
    "desarmes",
    "intercecoes",
    "bloqueios",
    "cortes",
    "recuperacoes",
    "pressoes",
)

FEET_EN: tuple[str, ...] = ("left-footed", "right-footed", "two-footed")
FEET_PT: tuple[str, ...] = ("canhotos", "destros", "ambidestros")

# The women's competitions, where date of birth, foot and height are all unknown. A
# question crossing one of these with any of those three has an answer the assistant can
# give only in part, and is supposed to say so.
WOMENS: tuple[str, ...] = (
    "Liga F",
    "the NWSL",
    "the FA Women's Super League",
    "the Frauen Bundesliga",
    "Serie A Women",
)

# Nothing in the seventeen per-90 columns measures any of these.
ABSENT_EN: tuple[str, ...] = (
    "aerial duels won",
    "saves",
    "yellow cards",
    "fouls committed",
    "offsides",
    "distance covered",
    "sprints",
    "crosses",
    "corners taken",
    "penalties saved",
    "expected assists",
    "transfer fees",
)
ABSENT_PT: tuple[str, ...] = (
    "duelos aereos ganhos",
    "defesas do guarda-redes",
    "cartoes amarelos",
    "faltas",
    "foras de jogo",
    "distancia percorrida",
    "cruzamentos",
    "cantos",
)

# Real people, genuinely absent from five men's leagues of 2015/16 and five women's of
# 2023/24 — most of them were children when this data was recorded.
ABSENT_PLAYERS: tuple[str, ...] = (
    "Erling Haaland",
    "Bruno Fernandes",
    "Jude Bellingham",
    "Vinicius Junior",
    "Bukayo Saka",
    "Pedri",
    "Jamal Musiala",
    "Rafael Leao",
    "Ruben Dias",
    "Joao Felix",
)


# --- the hand-written core ------------------------------------------------------------
#
# Kept apart from the generated ones because they were chosen to probe specific behaviour
# rather than to cover a grid: the phrasings a person actually types, the two questions
# with no requirement at all, and the ones that caught real bugs.
CURATED: tuple[Question, ...] = (
    Question("Who are the best left-footed wingers?", "constraint"),
    Question("Quais sao os melhores extremos canhotos?", "constraint"),
    Question("Find me a young centre back who is good on the ball.", "constraint"),
    Question("Quais os medios experientes com mais desarmes?", "constraint"),
    Question("Which two-footed midfielders move the ball forward best?", "constraint"),
    Question("Um lateral destro que faz a bola progredir", "constraint"),
    Question("Show me goalkeepers under 25.", "constraint"),
    Question("Guarda-redes experientes que recuperam muitas bolas", "constraint"),
    Question("Which wing-backs get forward the most?", "constraint"),
    Question("Alas jovens que conduzem muito a bola", "constraint"),
    Question("Two-footed centre backs, please.", "constraint"),
    Question("Medios ofensivos canhotos", "constraint"),
    Question("Which Brazilian forwards stand out?", "nationality"),
    Question("Que jogadoras brasileiras se destacam?", "nationality"),
    Question("Are there any Portuguese midfielders in the data?", "nationality"),
    Question("Avancados argentinos com bom remate", "nationality"),
    Question("Which French defenders are the most active?", "nationality"),
    Question("Extremos holandeses que driblam muito", "nationality"),
    Question("Any Senegalese midfielders worth a look?", "nationality"),
    Question("Jogadoras suecas que se destacam", "nationality"),
    Question("Which Italian defenders read the game best?", "nationality"),
    Question("Compare Neymar and Griezmann on dribbling.", "comparison"),
    Question("Compara o Messi e o Cristiano Ronaldo", "comparison"),
    Question("Who creates more, Iniesta or Griezmann?", "comparison"),
    Question("Quem e melhor a desarmar, Casimiro ou Kante?", "comparison"),
    Question("How good is Alexia Putellas?", "comparison"),
    Question("Fala-me do Bernd Leno", "comparison"),
    Question("Bale or Bakambu for goals?", "comparison"),
    Question("Compara a Alexia Putellas com a Caroline Graham Hansen", "comparison"),
    Question("Is Toni Kroos still progressing the ball?", "comparison"),
    Question("O Ibrahimovic remata muito?", "comparison"),
    Question("Who has the highest xG per 90 among centre forwards?", "metric"),
    Question("Quais os guarda-redes que mais recuperam bolas?", "metric"),
    Question("Which defenders make the most interceptions?", "metric"),
    Question("Quem tem mais passes progressivos entre os medios?", "metric"),
    Question("Who attempts the most dribbles?", "metric"),
    Question("Que avancados rematam mais vezes por 90 minutos?", "metric"),
    Question("Which midfielders press the most?", "metric"),
    Question("Which players complete the most passes?", "metric"),
    Question("Quem faz mais cortes por 90 minutos?", "metric"),
    Question("Who blocks the most shots?", "metric"),
    Question("Que jogadores conduzem mais a bola?", "metric"),
    Question("Which forwards create the most assists?", "metric"),
    Question("Which young players stand out in Liga F?", "coverage-gap"),
    Question("Extremos canhotos da NWSL", "coverage-gap"),
    Question("How old is Alexia Putellas?", "coverage-gap"),
    Question("Guarda-redes jovens da Frauen Bundesliga", "coverage-gap"),
    # No requirement at all, which is its own test: the assistant has to choose a reading
    # rather than refuse or invent a filter.
    Question("Who should I sign?", "vague"),
    Question("Quem e o melhor jogador do dataset?", "vague"),
    Question("Which players win the most aerial duels?", "unanswerable-metric"),
    Question("How many saves do the goalkeepers make?", "unanswerable-metric"),
    Question("Qual e a distancia percorrida por jogo?", "unanswerable-metric"),
    Question("Quantos cartoes amarelos tem o Sergio Ramos?", "unanswerable-metric"),
    Question("What is each player's market value?", "unanswerable-metric"),
    Question("Quao bom e o Otavio do FC Porto?", "unanswerable-player"),
    Question("How good is Erling Haaland?", "unanswerable-player"),
    Question("Fala-me do Bruno Fernandes no Sporting", "unanswerable-player"),
    Question("Quantos golos marcou o Benfica esta epoca?", "unanswerable-player"),
)


def generated(seed: int = 42) -> list[Question]:
    """Templates crossed with the vocabularies above, shuffled once and deterministically.

    Shuffled across the whole list rather than emitted in blocks, so a run stopped early by
    a quota still covers every category instead of finishing one and never reaching the
    others.
    """
    rng = random.Random(seed)
    out: list[Question] = []

    def add(text: str, kind: str) -> None:
        out.append(Question(text, kind))

    # Foot and position: the two constraints the parser enforces before ranking.
    for foot, role in zip(rng.choices(FEET_EN, k=18), rng.choices(ROLES_EN, k=18), strict=True):
        add(f"Which {foot} {role} stand out?", "constraint")
    for foot, role in zip(rng.choices(FEET_PT, k=18), rng.choices(ROLES_PT, k=18), strict=True):
        add(f"Quais os melhores {role} {foot}?", "constraint")

    # Age, a filter for the men and unknown for the women.
    for role in rng.sample(ROLES_EN, 6):
        add(f"Show me {role} under 23.", "constraint")
    for role in rng.sample(ROLES_PT, 6):
        add(f"Quais os {role} com mais de 32 anos?", "constraint")

    # Height, which lives in the profile prose rather than in any filter.
    for role in rng.sample(ROLES_EN, 5):
        add(f"Which are the tallest {role}?", "physical")
    for role in rng.sample(ROLES_PT, 5):
        add(f"Quais os {role} mais altos?", "physical")
    for name in rng.sample(PINNABLE, 6):
        add(f"How tall is {name}?", "physical")
    for name in rng.sample(PINNABLE, 5):
        add(f"Que idade tem o {name}?", "physical")

    # Nationality crossed with position.
    for adjective, role in zip(
        rng.choices(NATIONS_EN, k=16), rng.choices(ROLES_EN, k=16), strict=True
    ):
        add(f"Which {adjective} {role} are worth a look?", "nationality")
    for nation, role in zip(
        rng.choices(NATIONS_PT, k=15), rng.choices(ROLES_PT, k=15), strict=True
    ):
        add(f"Que {role} {nation} se destacam?", "nationality")

    # A named metric, which the prose profile may not mention at all.
    for metric, role in zip(
        rng.choices(METRICS_EN, k=16), rng.choices(ROLES_EN, k=16), strict=True
    ):
        add(f"Which {role} make the most {metric}?", "metric")
    for metric, role in zip(
        rng.choices(METRICS_PT, k=15), rng.choices(ROLES_PT, k=15), strict=True
    ):
        add(f"Que {role} tem mais {metric} por 90 minutos?", "metric")

    # Named players, singly and in pairs.
    for name in rng.sample(PINNABLE, 10):
        add(f"How good is {name}?", "comparison")
    pairs = rng.sample(PINNABLE, 16)
    for first, second in zip(pairs[::2], pairs[1::2], strict=True):
        add(f"Compare {first} and {second}.", "comparison")
    for name in rng.sample(PINNABLE, 6):
        add(f"Fala-me do {name}", "comparison")

    # Answerable only in part: those three attributes are unknown for every woman.
    for league in WOMENS:
        add(f"Which young players stand out in {league}?", "coverage-gap")
    for league in rng.sample(WOMENS, 4):
        add(f"Who are the left-footed players in {league}?", "coverage-gap")
    for league in rng.sample(WOMENS, 3):
        add(f"How tall are the defenders in {league}?", "coverage-gap")

    # No answer at all.
    for metric in ABSENT_EN:
        add(f"Which players lead for {metric}?", "unanswerable-metric")
    for metric in ABSENT_PT:
        add(f"Quem tem mais {metric}?", "unanswerable-metric")
    for name in ABSENT_PLAYERS:
        add(f"How good is {name}?", "unanswerable-player")

    rng.shuffle(out)
    return out


def all_questions(seed: int = 42) -> tuple[Question, ...]:
    """The curated core first, then the generated ones.

    Curated first on purpose: a run cut short by a quota then holds the questions chosen to
    probe specific behaviour, which are the ones worth having when only part of the set
    could be scored.

    Deduplicated by text. The templates draw with replacement, so "Which left-footed
    wingers stand out?" can come up twice, and a repeated question is a repeated LLM call
    that also double-weights whatever it happens to score.
    """
    seen: set[str] = set()
    out: list[Question] = []
    for question in (*CURATED, *generated(seed)):
        if question.text not in seen:
            seen.add(question.text)
            out.append(question)
    return tuple(out)
