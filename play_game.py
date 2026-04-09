#!/usr/bin/env python3
"""Language Games — Semantic word games powered by word vectors.

Uses GloVe vectors from Stanford NLP, loaded with a lightweight pure-numpy
backend.  Models are downloaded and cached automatically on first run.
"""

import os
import random
import string
import sys
import time
import urllib.request
import zipfile
from collections import Counter

import numpy as np
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
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
GLOVE_ZIP_SIZE = 862_182_613

MODELS = {
    "1": (50, "GloVe 50d  \u2014 light & fast"),
    "2": (100, "GloVe 100d \u2014 good balance (recommended)"),
    "3": (200, "GloVe 200d \u2014 higher quality"),
    "4": (300, "GloVe 300d \u2014 best quality"),
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
    "bright_cyan", "bright_magenta", "bright_yellow", "bright_green",
    "bright_red", "bright_blue", "deep_pink1", "dark_orange",
]

DIFFICULTY_PRESETS = {
    "easy":   {"chain_thresh": 0.20, "sprint_thresh": 0.20, "hot_guesses": 15, "hints": 30},
    "medium": {"chain_thresh": 0.30, "sprint_thresh": 0.30, "hot_guesses": 10, "hints": 20},
    "hard":   {"chain_thresh": 0.40, "sprint_thresh": 0.40, "hot_guesses": 7,  "hints": 12},
}

TEMP_LABELS = [
    (0.10, "\U0001f9ca", "Frozen",    "bright_blue"),
    (0.20, "\u2744\ufe0f",  "Freezing",  "blue"),
    (0.30, "\U0001f30a", "Cold",      "cyan"),
    (0.40, "\U0001f4a7", "Cool",      "dim cyan"),
    (0.50, "\U0001f324\ufe0f",  "Lukewarm",  "yellow"),
    (0.60, "\u2600\ufe0f",  "Warm",      "bright_yellow"),
    (0.70, "\U0001f525", "Hot",       "dark_orange"),
    (0.85, "\U0001f4a5", "Scorching", "bright_red"),
    (1.01, "\U0001f30b", "ON FIRE!",  "bold bright_red"),
]

CELEBRATION_FRAMES = [
    "\u2728 \U0001f3c6 \u2728",
    "\U0001f389 \U0001f3c6 \U0001f389",
    "\u2b50 \U0001f3c6 \u2b50",
    "\U0001f38a \U0001f3c6 \U0001f38a",
]

# ---------------------------------------------------------------------------
# Rich console
# ---------------------------------------------------------------------------

console = Console()

# ---------------------------------------------------------------------------
# WordVectors \u2014 lightweight, pure-numpy word-embedding store
# ---------------------------------------------------------------------------


class WordVectors:
    """Lightweight word-vector store backed by numpy."""

    def __init__(self, words: list[str], vectors: np.ndarray):
        self.words = words
        self._vectors = vectors
        self._index: dict[str, int] = {w: i for i, w in enumerate(words)}
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._unit = vectors / norms

    def __contains__(self, word: str) -> bool:
        return word in self._index

    @property
    def key_to_index(self) -> dict[str, int]:
        return self._index

    def get_vector(self, word: str) -> np.ndarray | None:
        idx = self._index.get(word)
        return self._unit[idx] if idx is not None else None

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
        sims[idx] = -2.0
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

    def analogy(
        self, positive: list[str], negative: list[str], topn: int = 5
    ) -> list[tuple[str, float]]:
        """Vector arithmetic: sum(positive) - sum(negative)."""
        dim = self._unit.shape[1]
        vec = np.zeros(dim, dtype=np.float32)
        exclude: set[int] = set()
        for w in positive:
            idx = self._index.get(w)
            if idx is not None:
                vec += self._unit[idx]
                exclude.add(idx)
        for w in negative:
            idx = self._index.get(w)
            if idx is not None:
                vec -= self._unit[idx]
                exclude.add(idx)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        sims = self._unit @ vec
        for idx in exclude:
            sims[idx] = -2.0
        if topn < len(sims):
            top = np.argpartition(sims, -topn)[-topn:]
        else:
            top = np.arange(len(sims))
        top = top[np.argsort(-sims[top])]
        return [(self.words[i], float(sims[i])) for i in top]

    def nearest_to_vec(self, vec: np.ndarray, topn: int = 5, exclude: set[str] | None = None) -> list[tuple[str, float]]:
        """Find nearest words to an arbitrary vector."""
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        sims = self._unit @ vec
        if exclude:
            for w in exclude:
                idx = self._index.get(w)
                if idx is not None:
                    sims[idx] = -2.0
        if topn < len(sims):
            top = np.argpartition(sims, -topn)[-topn:]
        else:
            top = np.arange(len(sims))
        top = top[np.argsort(-sims[top])]
        return [(self.words[i], float(sims[i])) for i in top]


# ---------------------------------------------------------------------------
# Model downloading / caching
# ---------------------------------------------------------------------------


def _ensure_glove_zip() -> str:
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

        tmp = zip_path + ".part"
        urllib.request.urlretrieve(GLOVE_ZIP_URL, tmp, reporthook=_hook)
        os.rename(tmp, zip_path)

    console.print("[green]Download complete.[/green]\n")
    return zip_path


def load_glove(dim: int = 100) -> WordVectors:
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
        del rows
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(
            npz_path, words=np.array(words, dtype=object), vectors=vectors
        )

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
difficulty: dict = DIFFICULTY_PRESETS["medium"]
session_wins: dict[int, int] = {}  # player_index -> total wins across games

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
        box=box.ROUNDED, border_style="dim cyan",
        title="[bold]Word-Vector Models[/bold]", title_style="cyan",
        padding=(0, 2),
    )
    table.add_column("#", style="bold cyan", width=3, justify="center")
    table.add_column("Model", style="bold white")
    table.add_column("Details", style="dim")
    for key, (dim, desc) in MODELS.items():
        table.add_row(key, f"glove.6B.{dim}d", desc)
    console.print(Align.center(table))
    console.print(
        "\n[dim]Vectors download once (~822 MB) and are cached in ~/.cache/language-games/[/dim]"
    )
    choice = Prompt.ask(
        "\n[bold cyan]Select a model[/bold cyan]",
        choices=list(MODELS.keys()), default="2",
    )
    dim, _ = MODELS[choice]
    model = load_glove(dim)


def choose_difficulty():
    global difficulty
    table = Table(box=box.SIMPLE, border_style="dim", padding=(0, 2))
    table.add_column("Level", style="bold")
    table.add_column("Description", style="dim")
    table.add_row("[green]easy[/green]", "Relaxed thresholds, more hints, more guesses")
    table.add_row("[yellow]medium[/yellow]", "Balanced (default)")
    table.add_row("[red]hard[/red]", "Strict thresholds, fewer hints, fewer guesses")
    console.print(table)
    choice = Prompt.ask(
        "[bold]Difficulty[/bold]",
        choices=["easy", "medium", "hard"], default="medium",
    )
    difficulty = DIFFICULTY_PRESETS[choice]
    console.print()


def load_words(topic_word: str | None = None, num_words: int = 10000):
    global english_words, english_words_list
    assert model is not None
    if topic_word:
        try:
            sims = model.most_similar(topic_word, topn=num_words)
            english_words = {w for w, _ in sims} | {topic_word}
        except KeyError:
            console.print(
                f"[yellow]'{topic_word}' not in vocabulary \u2014 using full dictionary[/yellow]"
            )
            english_words = {w for w in dictionary_words if w in model}
    else:
        with console.status("[dim]Filtering dictionary against model vocabulary...[/dim]"):
            english_words = {w for w in dictionary_words if w in model}
    english_words_list = list(english_words)
    console.print(f"[green]Ready[/green] \u2014 {len(english_words):,} playable words\n")


# ---------------------------------------------------------------------------
# Animation utilities
# ---------------------------------------------------------------------------


def animate_bar(score: float, width: int = 20, prefix: str = "    "):
    """Animated similarity bar that fills left-to-right."""
    clamped = max(0.0, min(1.0, score))
    target = int(clamped * width)
    try:
        with Live("", console=console, refresh_per_second=30, transient=True) as live:
            for i in range(target + 1):
                frac = i / max(target, 1)
                c = sim_color(score * frac if target > 0 else score)
                bar = "\u2588" * i + "\u2591" * (width - i)
                live.update(
                    Text(f"{prefix}{bar} {score:+.4f}", style=c)
                )
                time.sleep(0.018)
    except Exception:
        pass
    # Print final static version
    console.print(f"{prefix}{sim_bar(score)}")


def animate_countdown(n: int = 3):
    """Countdown before a reveal."""
    try:
        with Live("", console=console, refresh_per_second=4, transient=True) as live:
            for i in range(n, 0, -1):
                dots = "\u2022" * i
                live.update(Text(f"  {dots} {i} {dots}", style="bold yellow"))
                time.sleep(0.6)
    except Exception:
        pass


def animate_reveal(label: str, word: str, style: str = "bold white on magenta"):
    """Dramatic word reveal with typing effect."""
    animate_countdown(3)
    try:
        with Live("", console=console, refresh_per_second=20, transient=True) as live:
            displayed = ""
            for ch in word:
                displayed += ch
                live.update(
                    Text(f"  {label} {displayed}\u2588", style="dim")
                )
                time.sleep(0.05)
    except Exception:
        pass
    console.print(f"  {label} [{style}] {word} [/{style}]")


def celebrate(winner_idx: int):
    """Sparkle celebration for the winner."""
    tag = player_tag(winner_idx)
    try:
        with Live("", console=console, refresh_per_second=8, transient=True) as live:
            for _ in range(2):
                for frame in CELEBRATION_FRAMES:
                    live.update(
                        Text.from_markup(f"\n  {frame}  {tag} wins!\n")
                    )
                    time.sleep(0.2)
    except Exception:
        pass
    console.print(f"\n  {CELEBRATION_FRAMES[0]}  {tag} wins!\n")


def thermometer(score: float) -> str:
    """Return a temperature string for Hot & Cold."""
    for threshold, emoji, label, style in TEMP_LABELS:
        if score < threshold:
            clamped = max(0.0, min(1.0, score))
            width = 20
            filled = int(clamped * width)
            bar = "\u2588" * filled + "\u2591" * (width - filled)
            return f"{emoji} [{style}]{bar} {score:.3f}  {label}[/{style}]"
    # Fallback
    return f"\U0001f525 [bold red]{score:.3f}[/bold red]"


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
            console.print(f"  [red]'{w}' not in vocabulary \u2014 try again.[/red]")
        else:
            return w


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


# ---------------------------------------------------------------------------
# Score display
# ---------------------------------------------------------------------------


def show_scores(
    scores: list, num_players: int,
    label: str = "Score", fmt: str = ".3f",
):
    console.print()
    console.print(Rule("Final Scores", style="bold yellow"))
    console.print()

    table = Table(box=box.HEAVY_EDGE, border_style="yellow")
    table.add_column("Player", justify="center", style="bold")
    table.add_column(label, justify="center")
    table.add_column("", justify="center", width=4)

    best = max(scores)
    winner_idx = scores.index(best)
    for p in range(num_players):
        s = f"{scores[p]:{fmt}}" if isinstance(scores[p], float) else str(scores[p])
        winner = scores[p] == best
        table.add_row(
            player_tag(p),
            f"[bold green]{s}[/bold green]" if winner else s,
            "\U0001f451" if winner else "",
        )

    console.print(Align.center(table))

    # Track session wins and celebrate
    session_wins.setdefault(winner_idx, 0)
    session_wins[winner_idx] = session_wins.get(winner_idx, 0) + 1
    celebrate(winner_idx)


def show_session_leaderboard(num_players: int):
    """Show cumulative wins across all games this session."""
    if not session_wins:
        return
    console.print(Rule("Session Leaderboard", style="bold bright_white"))
    console.print()
    table = Table(box=box.DOUBLE_EDGE, border_style="bright_white")
    table.add_column("Player", justify="center", style="bold")
    table.add_column("Games Won", justify="center")
    for p in range(num_players):
        wins = session_wins.get(p, 0)
        table.add_row(player_tag(p), f"[bold]{wins}[/bold]")
    console.print(Align.center(table))
    console.print()


# ===================================================================
# Game 1 \u2014 Competitive Word Guessing
# ===================================================================


def game_1(num_turns: int = 5, num_players: int = 2):
    assert model is not None
    console.print(Panel(
        "[bold]Competitive Word Guessing[/bold]\n\n"
        "Each round a secret word is chosen.  You see a list of semantic hints.\n"
        "Type a word you think is close to the secret \u2014 similarity is your score.\n"
        "Scores [bold]accumulate[/bold] across rounds.",
        title="Game 1", border_style="magenta", padding=(1, 3),
    ))

    scores = [0.0] * num_players
    hint_count = difficulty["hints"]

    for turn in range(1, num_turns + 1):
        secret = rand_word()
        try:
            hints = [w for w, _ in model.most_similar(secret, topn=100)]
        except KeyError:
            continue

        console.print(Rule(f"Round {turn}/{num_turns}", style="magenta"))
        console.print()

        random.shuffle(hints)
        hint_str = "  ".join(f"[dim]{h}[/dim]" for h in hints[:hint_count])
        console.print(Panel(
            hint_str, title="[bold]Hints[/bold]",
            border_style="dim magenta", padding=(1, 2),
        ))
        console.print()

        for p in range(num_players):
            word = get_player_word(p, "guess the secret word")
            sim = float(model.similarity(word, secret))
            scores[p] += sim
            animate_bar(sim)
            console.print()

        animate_reveal("Secret word:", secret, "bold white on magenta")
        console.print()

    show_scores(scores, num_players, "Total Similarity")


# ===================================================================
# Game 2 \u2014 Closest Word Selection
# ===================================================================


def game_2(num_turns: int = 5, num_players: int = 2, num_words: int = 5):
    assert model is not None
    console.print(Panel(
        "[bold]Closest Word Selection[/bold]\n\n"
        "Pick the word from the list that is [bold]most similar[/bold] to the target.\n"
        "1 point per correct answer.",
        title="Game 2", border_style="blue", padding=(1, 3),
    ))

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
        console.print(f"  Target: [bold white on blue] {target} [/bold white on blue]\n")
        for i, w in enumerate(candidates, 1):
            console.print(f"    [bold cyan]{i}.[/bold cyan] {w}")
        console.print()

        for p in range(num_players):
            idx = _pick_number(p, len(candidates))
            if candidates[idx] == answer:
                scores[p] += 1
                console.print(f"    [green]Correct![/green]")
            else:
                console.print(f"    [red]Wrong.[/red]")
            console.print()

        console.print(f"  [bold blue]Answer:[/bold blue] [bold]{answer}[/bold]")
        for w in candidates:
            sim = float(model.similarity(target, w))
            mark = " [bold green]\u25c4[/bold green]" if w == answer else ""
            console.print(f"    {w:<20s} {sim_bar(sim)}{mark}")
        console.print()

    show_scores(scores, num_players, "Correct", fmt="d")


# ===================================================================
# Game 3 \u2014 Odd One Out
# ===================================================================


def game_3(num_turns: int = 5, num_players: int = 2, num_words: int = 5):
    assert model is not None
    console.print(Panel(
        "[bold]Odd One Out[/bold]\n\n"
        "One word in the list doesn't belong.  Find the [bold]semantic outlier[/bold].\n"
        "1 point per correct answer.",
        title="Game 3", border_style="green", padding=(1, 3),
    ))

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

        console.print(f"  [bold green]Outlier:[/bold green] [bold]{outlier}[/bold]\n")

    show_scores(scores, num_players, "Correct", fmt="d")


# ===================================================================
# Game 4 \u2014 Semantic Scrabble
# ===================================================================


def game_4(num_turns: int = 5, num_players: int = 2, num_chars: int = 10):
    assert model is not None
    console.print(Panel(
        "[bold]Semantic Scrabble[/bold]\n\n"
        "You get a target word and a rack of random letters.\n"
        "Form a real word from those letters that is as\n"
        "[bold]semantically similar[/bold] to the target as possible.\n"
        "Scores [bold]accumulate[/bold] across rounds.",
        title="Game 4", border_style="red", padding=(1, 3),
    ))

    scores = [0.0] * num_players

    for turn in range(1, num_turns + 1):
        target = rand_word()
        letters = [random.choice(string.ascii_lowercase) for _ in range(num_chars)]

        console.print(Rule(f"Round {turn}/{num_turns}", style="red"))
        console.print()
        console.print(f"  Target: [bold white on red] {target} [/bold white on red]\n")
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
                    console.print(f"  [red]'{word}' can't be formed from the letters.[/red]")
                    continue
                if not in_vocab(word):
                    console.print(f"  [red]'{word}' not in model vocabulary.[/red]")
                    continue
                if word not in dictionary_words:
                    console.print(f"  [red]'{word}' not a recognized English word.[/red]")
                    continue
                break
            sim = float(model.similarity(word, target))
            scores[p] += sim
            animate_bar(sim)
            console.print()

    show_scores(scores, num_players, "Total Similarity")


# ===================================================================
# Game 5 \u2014 Analogies
# ===================================================================


def game_5(num_turns: int = 5, num_players: int = 2):
    """A is to B as C is to ??? \u2014 classic word-vector analogy."""
    assert model is not None
    console.print(Panel(
        "[bold]Analogies[/bold]\n\n"
        "You'll see:  [italic]A is to B  as  C is to ???[/italic]\n"
        "The answer is computed with vector arithmetic.\n"
        "Guess the word \u2014 score = similarity to the computed answer.",
        title="Game 5", border_style="bright_cyan", padding=(1, 3),
    ))

    scores = [0.0] * num_players

    # Curated seed pairs that produce good analogies
    seed_pairs = [
        ("king", "queen"), ("man", "woman"), ("boy", "girl"),
        ("brother", "sister"), ("father", "mother"), ("uncle", "aunt"),
        ("husband", "wife"), ("prince", "princess"), ("son", "daughter"),
        ("big", "small"), ("hot", "cold"), ("fast", "slow"),
        ("good", "bad"), ("old", "young"), ("dark", "light"),
        ("france", "paris"), ("japan", "tokyo"), ("italy", "rome"),
        ("dog", "puppy"), ("cat", "kitten"), ("go", "went"),
        ("walk", "walking"), ("swim", "swimming"), ("teach", "teacher"),
    ]
    random.shuffle(seed_pairs)

    for turn in range(1, num_turns + 1):
        # Pick a seed pair and a third word
        if seed_pairs:
            a, b = seed_pairs.pop()
        else:
            a, b = random.choice([
                ("king", "queen"), ("man", "woman"), ("big", "small"),
            ])

        # Find a good C word: pick from similar words to A that aren't B
        try:
            c_candidates = [w for w, _ in model.most_similar(a, topn=50)
                           if w != b and w != a]
        except KeyError:
            continue
        if not c_candidates:
            continue
        c = random.choice(c_candidates[:20])

        # Compute answer: B - A + C
        results = model.analogy(positive=[b, c], negative=[a], topn=5)
        if not results:
            continue
        answer_word = results[0][0]

        console.print(Rule(f"Round {turn}/{num_turns}", style="bright_cyan"))
        console.print()
        console.print(
            f"  [bold bright_cyan]{a}[/bold bright_cyan] is to "
            f"[bold bright_cyan]{b}[/bold bright_cyan] as "
            f"[bold bright_cyan]{c}[/bold bright_cyan] is to [bold]???[/bold]"
        )
        console.print()

        for p in range(num_players):
            word = get_player_word(p, "your answer")
            sim = float(model.similarity(word, answer_word))
            scores[p] += sim
            exact = " [bold green]\u2714 Exact match![/bold green]" if word == answer_word else ""
            animate_bar(sim)
            if exact:
                console.print(exact)
            console.print()

        animate_reveal("Answer:", answer_word, "bold white on bright_cyan")
        # Show top 3 computed results
        console.print(f"  [dim]Top analogy results: {', '.join(w for w, _ in results[:3])}[/dim]")
        console.print()

    show_scores(scores, num_players, "Total Similarity")


# ===================================================================
# Game 6 \u2014 Word Chain
# ===================================================================


def game_6(num_turns: int = 10, num_players: int = 2):
    """Build a semantic chain \u2014 each word must relate to the previous one."""
    assert model is not None
    thresh = difficulty["chain_thresh"]
    console.print(Panel(
        f"[bold]Word Chain[/bold]\n\n"
        f"A starting word is given.  Take turns adding words.\n"
        f"Each word must have similarity [bold]> {thresh:.2f}[/bold] to the previous word.\n"
        f"No repeats!  You lose a \u2764\ufe0f if you break the chain.\n"
        f"Most \u2764\ufe0f remaining wins.",
        title="Game 6", border_style="bright_magenta", padding=(1, 3),
    ))

    lives = [3] * num_players
    chain: list[str] = []
    current_word = rand_word()
    chain.append(current_word)

    console.print(Rule("Chain Start", style="bright_magenta"))
    console.print(f"\n  Starting word: [bold white on bright_magenta] {current_word} [/bold white on bright_magenta]\n")

    turn = 0
    while turn < num_turns and any(l > 0 for l in lives):
        for p in range(num_players):
            if lives[p] <= 0:
                continue

            hearts = "[red]\u2764\ufe0f[/red] " * lives[p] + "[dim]\U0001f5a4[/dim] " * (3 - lives[p])
            console.print(f"  {player_tag(p)} {hearts}")
            console.print(f"  Current word: [bold]{current_word}[/bold]")

            word = get_player_word(p, "add to chain")

            if word in chain:
                console.print(f"    [red]'{word}' already used! \u2212\u2764\ufe0f[/red]")
                lives[p] -= 1
                console.print()
                continue

            sim = float(model.similarity(word, current_word))
            console.print(f"    Similarity: {sim_bar(sim)}")

            if sim < thresh:
                console.print(f"    [red]Below threshold ({thresh:.2f})! \u2212\u2764\ufe0f[/red]")
                lives[p] -= 1
            else:
                console.print(f"    [green]Chain extended![/green]")
                current_word = word
                chain.append(word)
            console.print()

        turn += 1

    # Show the chain
    console.print(Rule("Chain", style="dim bright_magenta"))
    chain_str = " \u2192 ".join(chain)
    console.print(Panel(chain_str, border_style="dim", padding=(0, 2)))

    show_scores(lives, num_players, "Lives Left", fmt="d")


# ===================================================================
# Game 7 \u2014 Hot & Cold
# ===================================================================


def game_7(num_turns: int = 3, num_players: int = 2):
    """Guess the secret word with temperature feedback."""
    assert model is not None
    max_guesses = difficulty["hot_guesses"]
    console.print(Panel(
        f"[bold]Hot & Cold[/bold]\n\n"
        f"A secret word is hidden.  You get [bold]{max_guesses} guesses[/bold].\n"
        f"After each guess you see a temperature reading \u2014 hotter = closer.\n"
        f"Score = best similarity achieved.  Exact match = bonus!",
        title="Game 7", border_style="dark_orange", padding=(1, 3),
    ))

    scores = [0.0] * num_players

    for turn in range(1, num_turns + 1):
        secret = rand_word()

        console.print(Rule(f"Round {turn}/{num_turns}", style="dark_orange"))
        console.print()
        console.print("  [bold]A secret word has been chosen...[/bold]\n")

        best_per_player = [0.0] * num_players
        guesses_used = [0] * num_players
        found = [False] * num_players

        for g in range(1, max_guesses + 1):
            console.print(f"  [dim]\u2500\u2500 Guess {g}/{max_guesses} \u2500\u2500[/dim]")
            for p in range(num_players):
                if found[p]:
                    console.print(f"  {player_tag(p)} [green]Already found it![/green]")
                    continue

                word = get_player_word(p, "guess")
                guesses_used[p] = g
                sim = float(model.similarity(word, secret))
                best_per_player[p] = max(best_per_player[p], sim)

                console.print(f"    {thermometer(sim)}")

                if word == secret:
                    console.print(f"    [bold green]\U0001f389 EXACT MATCH![/bold green]")
                    found[p] = True
                    best_per_player[p] = 1.5  # bonus for exact
                console.print()

            if all(found):
                break

        for p in range(num_players):
            scores[p] += best_per_player[p]

        animate_reveal("Secret word:", secret, "bold white on dark_orange")
        console.print()

    show_scores(scores, num_players, "Total Score")


# ===================================================================
# Game 8 \u2014 Category Sprint
# ===================================================================


def game_8(num_turns: int = 3, num_players: int = 2, words_per_turn: int = 5):
    """Name as many words in a category as you can."""
    assert model is not None
    thresh = difficulty["sprint_thresh"]
    console.print(Panel(
        f"[bold]Category Sprint[/bold]\n\n"
        f"A category word is given.  Name [bold]{words_per_turn} words[/bold] that belong.\n"
        f"Each must have similarity [bold]> {thresh:.2f}[/bold] to the category. No repeats.\n"
        f"Score = sum of similarities for valid words.",
        title="Game 8", border_style="bright_green", padding=(1, 3),
    ))

    scores = [0.0] * num_players

    for turn in range(1, num_turns + 1):
        category = rand_word()

        console.print(Rule(f"Round {turn}/{num_turns}", style="bright_green"))
        console.print()
        console.print(
            f"  Category: [bold white on bright_green] {category} [/bold white on bright_green]\n"
        )

        used_words: set[str] = set()

        for p in range(num_players):
            console.print(f"  {player_tag(p)} \u2014 name {words_per_turn} related words:\n")
            round_score = 0.0
            for w_num in range(1, words_per_turn + 1):
                while True:
                    word = Prompt.ask(
                        f"    [dim]({w_num}/{words_per_turn})[/dim] word"
                    ).strip().lower()
                    if not word:
                        console.print("    [red]Please enter a word.[/red]")
                        continue
                    if not in_vocab(word):
                        console.print(f"    [red]'{word}' not in vocabulary.[/red]")
                        continue
                    if word in used_words:
                        console.print(f"    [red]'{word}' already used this round![/red]")
                        continue
                    break

                used_words.add(word)
                sim = float(model.similarity(word, category))

                if sim >= thresh:
                    round_score += sim
                    console.print(f"      [green]\u2714[/green] {sim_bar(sim)}")
                else:
                    console.print(f"      [red]\u2718 Below threshold ({thresh:.2f})[/red] {sim_bar(sim)}")

            scores[p] += round_score
            console.print(f"    [dim]Round subtotal: {round_score:.3f}[/dim]\n")

    show_scores(scores, num_players, "Total Similarity")


# ===================================================================
# Main menu
# ===================================================================

GAME_TABLE_ROWS = [
    ("1", "Competitive Guessing", "Guess the secret word from semantic hints", "magenta"),
    ("2", "Closest Word",         "Pick the most similar word from a list",    "blue"),
    ("3", "Odd One Out",          "Find the word that doesn't belong",         "green"),
    ("4", "Semantic Scrabble",    "Form words from letters to match a meaning","red"),
    ("5", "Analogies",            "A is to B as C is to ???",                  "bright_cyan"),
    ("6", "Word Chain",           "Build a chain of related words",            "bright_magenta"),
    ("7", "Hot & Cold",           "Guess the secret with temperature hints",   "dark_orange"),
    ("8", "Category Sprint",      "Name words belonging to a category",        "bright_green"),
]


def show_menu():
    table = Table(box=box.ROUNDED, border_style="bright_white", padding=(0, 2))
    table.add_column("#", style="bold cyan", width=3, justify="center")
    table.add_column("Game", style="bold")
    table.add_column("Description", style="dim")
    for num, name, desc, color in GAME_TABLE_ROWS:
        table.add_row(f"[{color}]{num}[/{color}]", f"[{color}]{name}[/{color}]", desc)
    console.print(Align.center(table))
    console.print()


def main():
    console.clear()
    console.print(Panel(
        Align.center(Text(TITLE, style="bold cyan")),
        subtitle="[dim]Semantic word games powered by word vectors[/dim]",
        border_style="cyan", padding=(0, 4),
    ))

    load_dictionary()
    choose_and_load_model()

    last_players = 2

    while True:
        # Topic
        use_topic = Confirm.ask("[bold]Filter words by a topic?[/bold]", default=False)
        if use_topic:
            topic = Prompt.ask("  Enter a topic word").strip().lower()
            load_words(topic_word=topic)
        else:
            load_words()

        # Difficulty
        choose_difficulty()

        # Game menu
        show_menu()
        game = Prompt.ask(
            "[bold cyan]Choose a game[/bold cyan]",
            choices=[str(i) for i in range(1, 9)],
        )
        turns = IntPrompt.ask("[bold]Rounds[/bold]", default=5 if game not in ("7", "8") else 3)
        last_players = IntPrompt.ask("[bold]Players[/bold]", default=last_players)

        if game == "1":
            game_1(turns, last_players)
        elif game == "2":
            n = IntPrompt.ask("[bold]Words per round[/bold]", default=5)
            game_2(turns, last_players, n)
        elif game == "3":
            n = IntPrompt.ask("[bold]Words per round[/bold]", default=5)
            game_3(turns, last_players, n)
        elif game == "4":
            n = IntPrompt.ask("[bold]Letters per round[/bold]", default=10)
            game_4(turns, last_players, n)
        elif game == "5":
            game_5(turns, last_players)
        elif game == "6":
            game_6(turns, last_players)
        elif game == "7":
            game_7(turns, last_players)
        elif game == "8":
            n = IntPrompt.ask("[bold]Words per sprint[/bold]", default=5)
            game_8(turns, last_players, n)

        # Session leaderboard
        show_session_leaderboard(last_players)

        console.print()
        if not Confirm.ask("[bold]Play again?[/bold]", default=True):
            console.print("\n[dim]Thanks for playing![/dim]\n")
            break


if __name__ == "__main__":
    main()
