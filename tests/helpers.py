"""Shared fixtures-as-functions for the tests."""

from seance import game as G


def game_mid_round(seed: int = 5, rounds_done: int = 2, spoken: int = 2, allies=(0, 2)) -> G.Game:
    """A game with `rounds_done` finished rounds and `spoken` messages in the current one."""
    game = G.Game(G.deal(seed), allies=allies)
    for r in range(rounds_done):
        while (s := game.next_speaker()) is not None:
            game.submit_message(s, f"round {r + 1} words from seat {s}")
        game.submit_guesses((40, 50, 60, 70))
    for _ in range(spoken):
        s = game.next_speaker()
        game.submit_message(s, f"current words from seat {s}")
    return game


def notes_for(seat: int, rounds: int) -> list[dict[str, str]]:
    return [
        {"speak": f"seat {seat} speak note r{r + 1}", "guess": f"seat {seat} guess note r{r + 1}"}
        for r in range(rounds)
    ]
