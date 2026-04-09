#!/usr/bin/env python3
"""Language Games — Semantic word games powered by word vectors.

Uses GloVe vectors from Stanford NLP, loaded with a lightweight pure-numpy
backend.  Models are downloaded and cached automatically on first run.
"""

import os
import random
import string
import sys
import urllib.request
import zipfile
from collections import Counter

import numpy as np
from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORD_FILE = os.path.join(SCRIPT_DIR, "words_alpha.txt")

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "language-games")
GLOVE_ZIP_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_ZIP_SIZE = 862_182_613  # bytes (approximate)

MODELS = {
    "1": (50, "GloVe 50d  — light & fast"),
    "2": (100, "GloVe 100d — good balance (recommended)"),
    "3": (200, "GloVe 200d — higher quality"),
    "4": (300, "GloVe 300d — best quality"),
}

TITLE = r"""
  _
 | |   __ _ _ _  __ _ _  _ __ _ __ _ ___
 | |__/ _` | ' \/ _` | || / _` / _` / -_)
 |____\__,_|_||_\__, |\_,_\__,_\__, \___|
                |___/          |___/
   ___
  / __|__ _ _ __  ___ ___
 | (_ / _` | '  \/ -_|_-<
  \___\__,_|_|_|_\___/__/
"""

PLAYER_COLORS = [
    "bright_cyan",
    "bright_magenta",
    "bright_yellow",
    "bright_green",
    "bright_red",
    "bright_blue",
    "deep_pink1",
    "dark_orange",
]

# ---------------------------------------------------------------------------
# Rich console (global)
# ---------------------------------------------------------------------------

console = Console()

# ---------------------------------------------------------------------------
# WordVectors — lightweight, pure-numpy word-embedding store
# ---------------------------------------------------------------------------


class WordVectors:
    """Lightweight word-vector store backed by numpy."""

    def __init__(self, words: list[str], vectors: np.ndarray):
        self.words = words
        self._vectors = vectors  # (vocab, dim), float32
        self._index: dict[str, int] = {w: i for i, w in enumerate(words)}
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._unit = vectors / norms  # L2-normalised

    # -- membership / indexing ------------------------------------------

    def __contains__(self, word: str) -> bool:
        return word in self._index

    @property
    def key_to_index(self) -> dict[str, int]:
        return self._index

    # -- similarity operations ------------------------------------------

    def similarity(self, w1: str, w2: str) -> float:
        i, j = self._index.get(w1), self._index.get(w2)
        if i is None or j is None:
            return 0.0
        return float(self._unit[i] @ self._unit[j])

    def most_similar(self, word: str, topn: int = 10) -> list[tuple[str, float]]:
        idx = self._index.get(word)
        if idx is None:
            raise KeyError(word)
        sims = self._unit @ self._unit[idx]
        sims[idx] = -2.0  # exclude the query word itself
        if topn < len(sims):
            top = np.argpartition(sims, -topn)[-topn:]
        else:
            top = np.arange(len(sims))
        top = top[np.argsort(-sims[top])]
        return [(self.words[i], float(sims[i])) for i in top]

    def most_similar_to_given(self, word: str, candidates: list[str]) -> str:
        idx = self._index.get(word)
        if idx is None:
            raise KeyError(word)
        best, best_sim = None, -2.0
        for w in candidates:
            j = self._index.get(w)
            if j is not None:
                s = float(self._unit[idx] @ self._unit[j])
                if s > best_sim:
                    best, best_sim = w, s
        if best is None:
            raise KeyError("none of the candidates are in vocabulary")
        return best

    def doesnt_match(self, words: list[str]) -> str:
        idxs, valid = [], []
        for w in words:
            j = self._index.get(w)
            if j is not None:
                idxs.append(j)
                valid.append(w)
        if len(idxs) < 2:
            raise ValueError("need at least 2 words in vocabulary")
        vecs = self._unit[idxs]
        centroid = vecs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm
        return valid[int(np.argmin(vecs @ centroid))]


# ---------------------------------------------------------------------------
# Model downloading / caching
# ---------------------------------------------------------------------------


def _ensure_glove_zip() -> str:
    """Download glove.6B.zip if not already cached.  Returns path."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    zip_path = os.path.join(CACHE_DIR, "glove.6B.zip")
    if os.path.exists(zip_path):
        return zip_path

    console.print(
        "\n[bold]Downloading GloVe vectors from Stanford NLP[/bold] (~822 MB)"
    )
    console.print(f"[dim]{GLOVE_ZIP_URL}[/dim]\n")

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("glove.6B.zip", total=GLOVE_ZIP_SIZE)

        def _hook(block_num: int, block_size: int, total_size: int):
            if total_size > 0:
                progress.update(task, total=total_size)
            progress.update(task, completed=block_num * block_size)

        tmp_path = zip_path + ".part"
        urllib.request.urlretrieve(GLOVE_ZIP_URL, tmp_path, reporthook=_hook)
        os.rename(tmp_path, zip_path)

    console.print("[green]Download complete.[/green]\n")
    return zip_path


def load_glove(dim: int = 100) -> WordVectors:
    """Load GloVe vectors for the requested dimensionality (cached)."""
    npz_path = os.path.join(CACHE_DIR, f"glove.6B.{dim}d.npz")

    if os.path.exists(npz_path):
        console.print(f"[dim]Loading cached vectors from {npz_path}[/dim]")
        with console.status("[bold green]Loading vectors...[/bold green]"):
            data = np.load(npz_path, allow_pickle=True)
            wv = WordVectors(list(data["words"]), data["vectors"])
        console.print(
            f"[green]Loaded[/green] GloVe {dim}d ({len(wv.words):,} words)\n"
        )
        return wv

    # Need to extract from zip first
    zip_path = _ensure_glove_zip()
    txt_name = f"glove.6B.{dim}d.txt"

    console.print(f"[dim]Extracting {txt_name} from zip...[/dim]")
    with console.status("[bold green]Parsing vectors (one-time)...[/bold green]"):
        words: list[str] = []
        rows: list[list[float]] = []
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(txt_name) as f:
                for line in f:
                    parts = line.decode("utf-8", errors="replace").rstrip().split(" ")
                    words.append(parts[0])
                    rows.append([float(x) for x in parts[1:]])
        vectors = np.array(rows, dtype=np.float32)
        del rows  # free memory

        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(npz_path, words=np.array(words, dtype=object), vectors=vectors)

    wv = WordVectors(words, vectors)
    console.print(
        f"[green]Loaded[/green] GloVe {dim}d ({len(wv.words):,} words)\n"
    )
    return wv


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

model: WordVectors | None = None
english_words: set[str] = set()
english_words_list: list[str] = []
dictionary_words: set[str] = set()

# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def player_tag(p: int) -> str:
    c = PLAYER_COLORS[p % len(PLAYER_COLORS)]
    return f"[bold {c}]Player {p + 1}[/bold {c}]"


def load_dictionary():
    global dictionary_words
    with open(WORD_FILE) as f:
        dictionary_words = {line.strip().lower() for line in f if line.strip()}


def choose_and_load_model():
    global model

    console.print()
    table = Table(
        box=box.ROUNDED,
        border_style="dim cyan",
        title="[bold]Word-Vector Models[/bold]",
        title_style="cyan",
        padding=(0, 2),
    )
    table.add_column("#", style="bold cyan", width=3, justify="center")
    table.add_column("Model", style="bold white")
    table.add_column("Details", style="dim")
    for key, (dim, desc) in MODELS.items():
        table.add_row(key, f"glove.6B.{dim}d", desc)
    console.print(Align.center(table))
    console.print()
    console.print(
        "[dim]Vectors download once (~822 MB) and are cached in ~/.cache/language-games/[/dim]"
    )

    choice = Prompt.ask(
        "\n[bold cyan]Select a model[/bold cyan]",
        choices=list(MODELS.keys()),
        default="2",
    )
    dim, _ = MODELS[choice]
    model = load_glove(dim)


def load_words(topic_word: str | None = None, num_words: int = 10000):
    global english_words, english_words_list
    assert model is not None

    if topic_word:
        try:
            sims = model.most_similar(topic_word, topn=num_words)
            english_words = {w for w, _ in sims} | {topic_word}
        except KeyError:
            console.print(
                f"[yellow]'{topic_word}' not in vocabulary — using full dictionary[/yellow]"
            )
            english_words = {w for w in dictionary_words if w in model}
    else:
        with console.status(
            "[dim]Filtering dictionary against model vocabulary...[/dim]"
        ):
            english_words = {w for w in dictionary_words if w in model}

    english_words_list = list(english_words)
    console.print(f"[green]Ready[/green] — {len(english_words):,} playable words\n")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def rand_word() -> str:
    return random.choice(english_words_list)


def rand_words(n: int = 5) -> list[str]:
    return random.sample(english_words_list, min(n, len(english_words_list)))


def sim_color(score: float) -> str:
    if score >= 0.65:
        return "green"
    if score >= 0.35:
        return "yellow"
    return "red"


def sim_bar(score: float, width: int = 20) -> str:
    clamped = max(0.0, min(1.0, score))
    filled = int(clamped * width)
    c = sim_color(score)
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    return f"[{c}]{bar} {score:+.4f}[/{c}]"


def valid_letters(word: str, letters: list[str]) -> bool:
    avail = Counter(letters)
    for ch, need in Counter(word).items():
        if avail.get(ch, 0) < need:
            return False
    return True


def in_vocab(word: str) -> bool:
    assert model is not None
    return word in model


def get_player_word(p: int, prompt: str = "Enter a word") -> str:
    while True:
        w = Prompt.ask(f"  {player_tag(p)} {prompt}").strip().lower()
        if not w:
            console.print("  [red]Please enter a word.[/red]")
        elif not in_vocab(w):
            console.print(
                f"  [red]'{w}' is not in the vocabulary — try again.[/red]"
            )
        else:
            return w


# ---------------------------------------------------------------------------
# Score display
# ---------------------------------------------------------------------------


def show_scores(
    scores: list,
    num_players: int,
    label: str = "Score",
    fmt: str = ".3f",
):
    console.print()
    console.print(Rule("Final Scores", style="bold yellow"))
    console.print()

    table = Table(box=box.HEAVY_EDGE, border_style="yellow")
    table.add_column("Player", justify="center", style="bold")
    table.add_column(label, justify="center")
    table.add_column("", justify="center", width=4)

    best = max(scores)
    for p in range(num_players):
        s = f"{scores[p]:{fmt}}" if isinstance(scores[p], float) else str(scores[p])
        winner = scores[p] == best
        table.add_row(
            player_tag(p),
            f"[bold green]{s}[/bold green]" if winner else s,
            "\U0001f451" if winner else "",
        )

    console.print(Align.center(table))
    console.print()


# ---------------------------------------------------------------------------
# Shared input helper
# ---------------------------------------------------------------------------


def _pick_number(p: int, n: int) -> int:
    while True:
        raw = Prompt.ask(f"  {player_tag(p)} pick (1\u2013{n})")
        try:
            idx = int(raw) - 1
            if 0 <= idx < n:
                return idx
        except ValueError:
            pass
        console.print(f"  [red]Enter a number between 1 and {n}.[/red]")


# ===================================================================
# Game 1 — Competitive Word Guessing
# ===================================================================


def game_1(num_turns: int = 5, num_players: int = 2):
    assert model is not None
    console.print(
        Panel(
            "[bold]Competitive Word Guessing[/bold]\n\n"
            "Each round a secret word is chosen.  You see a list of semantic hints.\n"
            "Type a word you think is close to the secret — similarity is your score.\n"
            "Scores [bold]accumulate[/bold] across rounds.",
            title="Game 1",
            border_style="magenta",
            padding=(1, 3),
        )
    )

    scores = [0.0] * num_players

    for turn in range(1, num_turns + 1):
        secret = rand_word()
        try:
            hints = [w for w, _ in model.most_similar(secret, topn=100)]
        except KeyError:
            continue

        console.print(Rule(f"Round {turn}/{num_turns}", style="magenta"))
        console.print()

        random.shuffle(hints)
        hint_str = "  ".join(f"[dim]{h}[/dim]" for h in hints[:25])
        console.print(
            Panel(
                hint_str,
                title="[bold]Hints[/bold]",
                border_style="dim magenta",
                padding=(1, 2),
            )
        )
        console.print()

        for p in range(num_players):
            word = get_player_word(p, "guess the secret word")
            sim = float(model.similarity(word, secret))
            scores[p] += sim
            console.print(f"    {sim_bar(sim)}")
            console.print()

        console.print(
            f"  Secret word: [bold white on magenta] {secret} [/bold white on magenta]\n"
        )

    show_scores(scores, num_players, "Total Similarity")


# ===================================================================
# Game 2 — Closest Word Selection
# ===================================================================


def game_2(num_turns: int = 5, num_players: int = 2, num_words: int = 5):
    assert model is not None
    console.print(
        Panel(
            "[bold]Closest Word Selection[/bold]\n\n"
            "Pick the word from the list that is [bold]most similar[/bold] to the target.\n"
            "1 point per correct answer.",
            title="Game 2",
            border_style="blue",
            padding=(1, 3),
        )
    )

    scores = [0] * num_players

    for turn in range(1, num_turns + 1):
        target = rand_word()
        candidates = [w for w in rand_words(num_words) if in_vocab(w)]
        if len(candidates) < 2:
            continue
        try:
            answer = model.most_similar_to_given(target, candidates)
        except KeyError:
            continue

        console.print(Rule(f"Round {turn}/{num_turns}", style="blue"))
        console.print()
        console.print(
            f"  Target: [bold white on blue] {target} [/bold white on blue]\n"
        )
        for i, w in enumerate(candidates, 1):
            console.print(f"    [bold cyan]{i}.[/bold cyan] {w}")
        console.print()

        for p in range(num_players):
            idx = _pick_number(p, len(candidates))
            chosen = candidates[idx]
            if chosen == answer:
                scores[p] += 1
                console.print(f"    [green]Correct![/green]")
            else:
                console.print(f"    [red]Wrong.[/red]")
            console.print()

        # reveal similarities
        console.print(f"  [bold blue]Answer:[/bold blue] [bold]{answer}[/bold]")
        for w in candidates:
            sim = float(model.similarity(target, w))
            mark = " [bold green]<[/bold green]" if w == answer else ""
            console.print(f"    {w:<20s} {sim_bar(sim)}{mark}")
        console.print()

    show_scores(scores, num_players, "Correct", fmt="d")


# ===================================================================
# Game 3 — Odd One Out
# ===================================================================


def game_3(num_turns: int = 5, num_players: int = 2, num_words: int = 5):
    assert model is not None
    console.print(
        Panel(
            "[bold]Odd One Out[/bold]\n\n"
            "One word in the list doesn't belong.  Find the [bold]semantic outlier[/bold].\n"
            "1 point per correct answer.",
            title="Game 3",
            border_style="green",
            padding=(1, 3),
        )
    )

    scores = [0] * num_players

    for turn in range(1, num_turns + 1):
        words = [w for w in rand_words(num_words) if in_vocab(w)]
        if len(words) < 3:
            continue
        try:
            outlier = model.doesnt_match(words)
        except ValueError:
            continue

        console.print(Rule(f"Round {turn}/{num_turns}", style="green"))
        console.print()
        console.print("  Which word [bold]doesn't belong[/bold]?\n")
        for i, w in enumerate(words, 1):
            console.print(f"    [bold cyan]{i}.[/bold cyan] {w}")
        console.print()

        for p in range(num_players):
            idx = _pick_number(p, len(words))
            if words[idx] == outlier:
                scores[p] += 1
                console.print(f"    [green]Correct![/green]")
            else:
                console.print(f"    [red]Wrong.[/red]")
            console.print()

        console.print(
            f"  [bold green]Outlier:[/bold green] [bold]{outlier}[/bold]\n"
        )

    show_scores(scores, num_players, "Correct", fmt="d")


# ===================================================================
# Game 4 — Semantic Scrabble
# ===================================================================


def game_4(
    num_turns: int = 5,
    num_players: int = 2,
    num_chars: int = 10,
):
    assert model is not None
    console.print(
        Panel(
            "[bold]Semantic Scrabble[/bold]\n\n"
            "You get a target word and a rack of random letters.\n"
            "Form a real word from those letters that is as\n"
            "[bold]semantically similar[/bold] to the target as possible.\n"
            "Scores [bold]accumulate[/bold] across rounds.",
            title="Game 4",
            border_style="red",
            padding=(1, 3),
        )
    )

    scores = [0.0] * num_players

    for turn in range(1, num_turns + 1):
        target = rand_word()
        letters = [random.choice(string.ascii_lowercase) for _ in range(num_chars)]

        console.print(Rule(f"Round {turn}/{num_turns}", style="red"))
        console.print()
        console.print(
            f"  Target: [bold white on red] {target} [/bold white on red]\n"
        )
        tiles = "  ".join(
            f"[bold white on dark_red] {l.upper()} [/bold white on dark_red]"
            for l in letters
        )
        console.print(f"  {tiles}\n")

        for p in range(num_players):
            while True:
                word = Prompt.ask(f"  {player_tag(p)} form a word").strip().lower()
                if not word:
                    console.print("  [red]Please enter a word.[/red]")
                    continue
                if not valid_letters(word, letters):
                    console.print(
                        f"  [red]'{word}' can't be formed from the available letters.[/red]"
                    )
                    continue
                if not in_vocab(word):
                    console.print(
                        f"  [red]'{word}' is not in the model vocabulary.[/red]"
                    )
                    continue
                if word not in dictionary_words:
                    console.print(
                        f"  [red]'{word}' is not a recognized English word.[/red]"
                    )
                    continue
                break

            sim = float(model.similarity(word, target))
            scores[p] += sim
            console.print(f"    {sim_bar(sim)}\n")

    show_scores(scores, num_players, "Total Similarity")


# ===================================================================
# Main menu
# ===================================================================

GAME_TABLE_ROWS = [
    ("1", "Competitive Guessing", "Guess the secret word from semantic hints"),
    ("2", "Closest Word", "Pick the most similar word from a list"),
    ("3", "Odd One Out", "Find the word that doesn't belong"),
    ("4", "Semantic Scrabble", "Form words from letters to match a meaning"),
]


def show_menu():
    table = Table(box=box.ROUNDED, border_style="bright_white", padding=(0, 2))
    table.add_column("#", style="bold cyan", width=3, justify="center")
    table.add_column("Game", style="bold")
    table.add_column("Description", style="dim")
    for row in GAME_TABLE_ROWS:
        table.add_row(*row)
    console.print(Align.center(table))
    console.print()


def main():
    console.clear()
    console.print(
        Panel(
            Align.center(Text(TITLE, style="bold cyan")),
            subtitle="[dim]Semantic word games powered by word vectors[/dim]",
            border_style="cyan",
            padding=(0, 4),
        )
    )

    load_dictionary()
    choose_and_load_model()

    while True:
        use_topic = Confirm.ask(
            "[bold]Filter words by a topic?[/bold]", default=False
        )
        if use_topic:
            topic = Prompt.ask("  Enter a topic word").strip().lower()
            load_words(topic_word=topic)
        else:
            load_words()

        show_menu()
        game = Prompt.ask(
            "[bold cyan]Choose a game[/bold cyan]", choices=["1", "2", "3", "4"]
        )
        turns = IntPrompt.ask("[bold]Rounds[/bold]", default=5)
        players = IntPrompt.ask("[bold]Players[/bold]", default=2)

        if game == "1":
            game_1(turns, players)
        elif game == "2":
            n = IntPrompt.ask("[bold]Words per round[/bold]", default=5)
            game_2(turns, players, n)
        elif game == "3":
            n = IntPrompt.ask("[bold]Words per round[/bold]", default=5)
            game_3(turns, players, n)
        elif game == "4":
            n = IntPrompt.ask("[bold]Letters per round[/bold]", default=10)
            game_4(turns, players, n)

        console.print()
        if not Confirm.ask("[bold]Play again?[/bold]", default=True):
            console.print("\n[dim]Thanks for playing![/dim]\n")
            break


if __name__ == "__main__":
    main()
