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


@pytest.mark.parametrize(
    "secret,guesses,expected",
    [
        (50, (50, 45, 30, 90), (3, 2, 1, 0)),
        (50, (90, 30, 45, 50), (0, 1, 2, 3)),
        (50, (45, 55, 30, 90), (2.5, 2.5, 1, 0)),  # two tied for closest, equal distance either side
        (50, (51, 51, 51, 10), (2, 2, 2, 0)),  # three tied for closest
        (50, (50, 40, 60, 99), (3, 1.5, 1.5, 0)),  # tie for second
        (50, (50, 40, 20, 80), (3, 2, 0.5, 0.5)),  # tie for last
        (50, (50, 40, 60, 40), (3, 1, 1, 1)),  # three tied for second
        (50, (40, 60, 40, 60), (1.5, 1.5, 1.5, 1.5)),  # all four tied
        (50, (49, 49, 70, 70), (2.5, 2.5, 0.5, 0.5)),  # two ties
        (1, (100, 1, 2, 3), (0, 3, 2, 1)),
    ],
)
def test_scoring_and_split_point_ties(secret, guesses, expected):
    got = G.score_round(secret, guesses)
    assert got == pytest.approx(expected)
    assert all(isinstance(p, float) for p in got)


def test_every_round_hands_out_exactly_six_points():
    import itertools

    for guesses in itertools.product((40, 45, 50, 55, 60), repeat=4):
        pts = G.score_round(50, guesses)
        assert sum(pts) == pytest.approx(6)
        assert all((2 * p) == int(2 * p) for p in pts)  # multiples of 0.5


def test_ties_never_reward_identical_guesses():
    # Two allies guessing the same number earn no more together than guessing
    # adjacent numbers on the same side would.
    same = G.score_round(50, (48, 48, 60, 70))
    apart = G.score_round(50, (48, 47, 60, 70))
    assert same[0] + same[1] == apart[0] + apart[1] == 5


def test_allies_pool_points():
    assert G.payoffs((10, 7, 5, 2), None) == (10, 7, 5, 2)
    assert G.payoffs((10, 7, 5, 2), (1, 3)) == (10, 9, 5, 9)


def _play_round(game, guesses):
    while game.posting_phase is not None:
        game.submit_posts([f"hello from {s}" for s in range(4)])
    return game.submit_guesses(guesses)


def test_game_enforces_simultaneous_posting_then_guessing():
    game = G.Game(G.deal(3), allies=(2, 0))
    assert game.allies == (0, 2)
    assert game.posting_phase == 0 and game.posts_so_far() == []
    with pytest.raises(RuntimeError):
        game.submit_guesses((1, 2, 3, 4))  # nobody has posted
    with pytest.raises(ValueError):
        game.submit_posts(["only", "three", "posts"])  # all four post at once
    with pytest.raises(ValueError):
        game.submit_posts(["ok", "word " * (G.MAX_MESSAGE_WORDS + 1), "ok", "ok"])
    assert game.posts_so_far() == []  # a rejected batch records nothing
    game.submit_posts(["a", "b", "c", "d"])
    assert game.posting_phase is None
    assert game.posts_so_far() == [("a", "b", "c", "d")]
    with pytest.raises(RuntimeError):
        game.submit_posts(["again"] * 4)  # one posting phase by default
    with pytest.raises(ValueError):
        game.submit_guesses((0, 50, 50, 50))  # 0 is out of range
    with pytest.raises(TypeError):
        game.submit_guesses((50.5, 50, 50, 50))
    res = game.submit_guesses((50, 50, 50, 50))
    assert res.points == (1.5, 1.5, 1.5, 1.5)
    assert res.posts == (("a", "b", "c", "d"),)
    assert game.posting_phase == 0 and game.posts_so_far() == []  # round 2 starts fresh


def test_two_posting_phases():
    game = G.Game(G.deal(3), posting_phases=2)
    game.submit_posts(["a1", "b1", "c1", "d1"])
    assert game.posting_phase == 1
    with pytest.raises(RuntimeError):
        game.submit_guesses((50, 50, 50, 50))  # second posting still open
    game.submit_posts(["a2", "b2", "c2", "d2"])
    res = game.submit_guesses((50, 50, 50, 50))
    assert res.posts == (("a1", "b1", "c1", "d1"), ("a2", "b2", "c2", "d2"))
    for bad in (0, 3):
        with pytest.raises(ValueError):
            G.Game(G.deal(3), posting_phases=bad)


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
