import collections

import pytest

from seance import game as G


@pytest.mark.parametrize("seed", range(200))
def test_every_vision_contains_the_secret_and_is_40_wide(seed):
    setup = G.deal(seed)
    assert len(setup.rounds) == G.N_ROUNDS
    for rnd in setup.rounds:
        assert G.SECRET_MIN <= rnd.secret <= G.SECRET_MAX
        assert len(rnd.visions) == G.N_MEDIUMS
        for v in rnd.visions:
            assert v.hi - v.lo + 1 == G.VISION_WIDTH
            assert v.contains(rnd.secret)


def test_secret_position_inside_vision_is_uniform_and_ranges_are_not_clipped():
    offsets = collections.Counter()
    below = above = 0
    for seed in range(2000):
        for rnd in G.deal(seed).rounds:
            for v in rnd.visions:
                offsets[rnd.secret - v.lo] += 1
                below += v.lo < G.SECRET_MIN
                above += v.hi > G.SECRET_MAX
    assert set(offsets) == set(range(G.VISION_WIDTH))
    n = sum(offsets.values())
    expected = n / G.VISION_WIDTH
    # 80,000 draws over 40 cells: each cell within 10% of expectation.
    assert all(abs(c - expected) < 0.1 * expected for c in offsets.values())
    assert below > 0 and above > 0


def test_deal_is_deterministic_per_seed():
    assert G.deal(7) == G.deal(7)
    assert G.deal(7) != G.deal(8)


def test_speaking_order_rotates_round_robin():
    setup = G.deal(1)
    assert setup.rounds[0].speaking_order == (0, 1, 2, 3)
    assert setup.rounds[1].speaking_order == (1, 2, 3, 0)
    assert setup.rounds[4].speaking_order == (0, 1, 2, 3)
    assert setup.rounds[9].speaking_order == (1, 2, 3, 0)


@pytest.mark.parametrize(
    "secret,guesses,expected",
    [
        (50, (50, 45, 30, 90), (3, 2, 1, 0)),
        (50, (90, 30, 45, 50), (0, 1, 2, 3)),
        (50, (45, 55, 30, 90), (3, 3, 1, 0)),  # tie for closest, equal distance either side
        (50, (50, 40, 60, 99), (3, 2, 2, 0)),  # tie for second
        (50, (50, 40, 20, 80), (3, 2, 1, 1)),  # tie for last
        (50, (50, 40, 60, 40), (3, 2, 2, 2)),  # three tied for second
        (50, (40, 60, 40, 60), (3, 3, 3, 3)),  # all tied
        (50, (51, 51, 51, 10), (3, 3, 3, 0)),
        (1, (100, 1, 2, 3), (0, 3, 2, 1)),
    ],
)
def test_scoring_and_ties(secret, guesses, expected):
    assert G.score_round(secret, guesses) == expected


def test_points_per_round_never_exceed_six_unless_tied():
    # Without ties the round hands out exactly 3+2+1+0.
    assert sum(G.score_round(50, (50, 48, 40, 1))) == 6


def test_allies_pool_points():
    assert G.payoffs((10, 7, 5, 2), None) == (10, 7, 5, 2)
    assert G.payoffs((10, 7, 5, 2), (1, 3)) == (10, 9, 5, 9)


def _play_round(game, guesses):
    while (s := game.next_speaker()) is not None:
        game.submit_message(s, f"hello from {s}")
    return game.submit_guesses(guesses)


def test_game_enforces_order_of_play():
    game = G.Game(G.deal(3), allies=(2, 0))
    assert game.allies == (0, 2)
    with pytest.raises(RuntimeError):
        game.submit_guesses((1, 2, 3, 4))  # nobody has spoken
    with pytest.raises(RuntimeError):
        game.submit_message(1, "out of turn")  # seat 0 speaks first in round 1
    game.submit_message(0, "first")
    with pytest.raises(ValueError):
        game.submit_message(1, "word " * (G.MAX_MESSAGE_WORDS + 1))
    for s in (1, 2, 3):
        game.submit_message(s, "ok")
    with pytest.raises(RuntimeError):
        game.submit_message(0, "again")
    with pytest.raises(ValueError):
        game.submit_guesses((0, 50, 50, 50))  # 0 is out of range
    with pytest.raises(TypeError):
        game.submit_guesses((50.5, 50, 50, 50))
    res = game.submit_guesses((50, 50, 50, 50))
    assert res.points == (3, 3, 3, 3)
    assert game.next_speaker() == 1  # round 2 starts with seat 1


def test_full_game_totals_and_payoffs():
    setup = G.deal(11)
    game = G.Game(setup, allies=(1, 2))
    for rnd in setup.rounds:
        _play_round(game, tuple(v.middle if 1 <= v.middle <= 100 else 50 for v in rnd.visions))
    assert game.finished
    totals = game.totals()
    assert totals == tuple(sum(r.points[i] for r in game.results) for i in range(4))
    pay = game.payoffs()
    assert pay[1] == pay[2] == totals[1] + totals[2]
    assert pay[0] == totals[0] and pay[3] == totals[3]
    with pytest.raises(RuntimeError):
        game.current_round


def test_bad_allies_rejected():
    with pytest.raises(ValueError):
        G.Game(G.deal(1), allies=(1, 1))
    with pytest.raises(ValueError):
        G.Game(G.deal(1), allies=(0, 4))
