# Language Games

Dead simple semantic word games powered by word vectors, played in the terminal.

Uses [GloVe](https://nlp.stanford.edu/projects/glove/) vectors from Stanford NLP with a lightweight pure-numpy backend — no heavy ML frameworks required. Models download automatically on first run and are cached for instant reloading.

## Games

| # | Game | How it works |
|---|------|-------------|
| 1 | **Competitive Guessing** | A secret word is chosen. You see semantic hints. Guess the word — cosine similarity is your score. |
| 2 | **Closest Word** | Pick the word from a list that is most similar to a target word. |
| 3 | **Odd One Out** | Find the word that doesn't belong in a group. |
| 4 | **Semantic Scrabble** | Form words from random letter tiles that are semantically close to a target. |

All games support any number of players. Scores accumulate across rounds.

## Install

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Or with pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python play_game.py
```

On first launch you choose a vector dimensionality. The GloVe zip (~822 MB) downloads once with a progress bar and is cached in `~/.cache/language-games/`. Subsequent launches load in seconds.

## Screenshots

### Title & Model Selection
<img src="screenshots/title.svg" alt="Title screen">

### Game Menu
<img src="screenshots/game_menu.svg" alt="Game menu">

### Game 1 — Competitive Guessing
<img src="screenshots/game1.svg" alt="Game 1 gameplay">

### Game 2 — Closest Word
<img src="screenshots/game2.svg" alt="Game 2 gameplay">

### Game 3 — Odd One Out
<img src="screenshots/game3.svg" alt="Game 3 gameplay">

### Game 4 — Semantic Scrabble
<img src="screenshots/game4.svg" alt="Game 4 gameplay">

### Final Scores
<img src="screenshots/scores.svg" alt="Final scores">

## How It Works

Words are represented as dense vectors in a high-dimensional space (50–300 dimensions) where semantically similar words are closer together. The games use cosine similarity between these vectors to measure how related two words are.

The `WordVectors` class is a lightweight, pure-numpy implementation that supports:
- **Cosine similarity** between any two words
- **Most-similar lookup** using fast `argpartition`
- **Outlier detection** by finding the word furthest from the group centroid

## Dependencies

Just `numpy` and `rich` — no compiled ML libraries needed.

## License

MIT
