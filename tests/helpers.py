"""Shared fixtures-as-functions for the tests."""

from seance import game as G


def game_mid_round(
    seed: int = 5, rounds_done: int = 2, posted: int = 0, allies=(0, 2), posting_phases: int = 1
) -> G.Game:
    """A game with `rounds_done` finished rounds and `posted` completed posting
    phases in the current one (posted == posting_phases means guess time)."""
    game = G.Game(G.deal(seed), allies=allies, posting_phases=posting_phases)
    for r in range(rounds_done):
        for k in range(posting_phases):
            game.submit_posts([f"round {r + 1} post {k + 1} words from seat {s}" for s in range(4)])
        game.submit_guesses((40, 50, 60, 70))
    for k in range(posted):
        game.submit_posts([f"current post {k + 1} words from seat {s}" for s in range(4)])
    return game


def notes_for(seat: int, rounds: int, posting_phases: int = 1) -> list[dict]:
    return [
        {
            "speak": [f"seat {seat} speak note r{r + 1} p{k + 1}" for k in range(posting_phases)],
            "guess": f"seat {seat} guess note r{r + 1}",
        }
        for r in range(rounds)
    ]
