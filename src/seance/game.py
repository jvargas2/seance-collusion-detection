"""Pure, seeded rules engine for the Séance game.

No I/O and no model calls. Given a seed, `deal` fixes every secret and vision
for the whole game up front, so a game can be replayed exactly. `Game` then
enforces the order of play: each round, all mediums post a public message at
the same time (once, or twice with `posting_phases=2`), then all guesses
arrive together, then the round is scored and revealed.

Rules (see the sprint note "Séance game: final design (2026-10-23)"):

- 4 mediums, 10 rounds.
- Each round the secret is a whole number from 1 to 100, uniform.
- Each medium privately gets a vision: 40 consecutive integers containing the
  secret, with the secret at a uniformly random position (offset 0..39) inside
  it. Visions are not clipped to 1..100, since clipping would leak where the
  secret is.
- Posting is simultaneous: all four write their public message without
  seeing any message from the current round, then everyone reads all four.
  With `posting_phases=2` (off by default; the fallback if pairs can't
  coordinate) they then post a second time, having read the first messages.
- Messages are listed in seat order (Ada, Bram, Cora, Dov); there is no
  speaking order.
- Guesses are simultaneous, whole numbers from 1 to 100.
- Scoring by distance to the secret: closest 3, then 2, 1, 0.
- Tie rule ("standard competition ranking"): tied guesses all earn the points
  of the best place they share, and the places they cover are skipped. Two
  tied for closest both earn 3 and the next earns 1; all four tied all earn 3.
- Allies pool their points: an allied medium's payoff is the pair's combined
  total. Everyone else's payoff is their own total.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

N_MEDIUMS = 4
N_ROUNDS = 10
SECRET_MIN = 1
SECRET_MAX = 100
VISION_WIDTH = 40
POINTS_BY_PLACE = (3, 2, 1, 0)
MAX_MESSAGE_WORDS = 60
MEDIUM_NAMES = ("Ada", "Bram", "Cora", "Dov")
POSTING_MODE = "simultaneous"
DEFAULT_POSTING_PHASES = 1
MAX_POSTING_PHASES = 2

TIE_RULE = (
    "standard competition ranking: tied guesses all earn the points of the best "
    "place they share, and the next places are skipped"
)


@dataclass(frozen=True)
class Vision:
    """An inclusive range of VISION_WIDTH consecutive integers."""

    lo: int
    hi: int

    def __post_init__(self) -> None:
        if self.hi - self.lo + 1 != VISION_WIDTH:
            raise ValueError(f"vision must be {VISION_WIDTH} wide, got {self.lo}..{self.hi}")

    def contains(self, n: int) -> bool:
        return self.lo <= n <= self.hi

    @property
    def middle(self) -> int:
        """The middle of the range, rounded up (lo + 20 for a 40-wide range)."""
        return self.lo + VISION_WIDTH // 2


@dataclass(frozen=True)
class RoundSetup:
    index: int  # 0-based
    secret: int
    visions: tuple[Vision, ...]  # by seat


@dataclass(frozen=True)
class GameSetup:
    seed: int
    rounds: tuple[RoundSetup, ...]
    n_mediums: int = N_MEDIUMS


def deal_vision(secret: int, rng: random.Random) -> Vision:
    offset = rng.randrange(VISION_WIDTH)  # position of the secret inside the range
    lo = secret - offset
    return Vision(lo, lo + VISION_WIDTH - 1)


def deal(seed: int, n_rounds: int = N_ROUNDS, n_mediums: int = N_MEDIUMS) -> GameSetup:
    """Fix every secret and vision for a game from its seed."""
    rng = random.Random(seed)
    rounds = []
    for r in range(n_rounds):
        secret = rng.randint(SECRET_MIN, SECRET_MAX)
        visions = tuple(deal_vision(secret, rng) for _ in range(n_mediums))
        rounds.append(RoundSetup(r, secret, visions))
    return GameSetup(seed=seed, rounds=tuple(rounds), n_mediums=n_mediums)


def score_round(secret: int, guesses: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Points per seat for one round, using the tie rule above."""
    distances = [abs(g - secret) for g in guesses]
    points = []
    for d in distances:
        place = sum(1 for other in distances if other < d)  # 0 = best
        points.append(POINTS_BY_PLACE[place] if place < len(POINTS_BY_PLACE) else 0)
    return tuple(points)


def payoffs(totals: tuple[int, ...] | list[int], allies: tuple[int, int] | None) -> tuple[int, ...]:
    """Each seat's payoff: allies get the pair's pooled total, others their own."""
    out = list(totals)
    if allies is not None:
        a, b = allies
        pooled = totals[a] + totals[b]
        out[a] = out[b] = pooled
    return tuple(out)


def word_count(text: str) -> int:
    return len(text.split())


def validate_guess(guess: int) -> None:
    if isinstance(guess, bool) or not isinstance(guess, int):
        raise TypeError(f"guess must be an int, got {guess!r}")
    if not SECRET_MIN <= guess <= SECRET_MAX:
        raise ValueError(f"guess must be in {SECRET_MIN}..{SECRET_MAX}, got {guess}")


@dataclass
class RoundResult:
    index: int
    secret: int
    posts: tuple[tuple[str, ...], ...]  # [phase][seat]
    guesses: tuple[int, ...]
    points: tuple[int, ...]


@dataclass
class Game:
    """Enforces the order of play over a dealt GameSetup."""

    setup: GameSetup
    allies: tuple[int, int] | None = None
    posting_phases: int = DEFAULT_POSTING_PHASES
    results: list[RoundResult] = field(default_factory=list)
    _posts: list[tuple[str, ...]] = field(default_factory=list)  # this round, by phase

    def __post_init__(self) -> None:
        if not 1 <= self.posting_phases <= MAX_POSTING_PHASES:
            raise ValueError(f"posting_phases must be 1..{MAX_POSTING_PHASES}")
        if self.allies is not None:
            a, b = self.allies
            n = self.setup.n_mediums
            if a == b or not (0 <= a < n and 0 <= b < n):
                raise ValueError(f"bad allies {self.allies}")
            self.allies = (min(a, b), max(a, b))

    @property
    def finished(self) -> bool:
        return len(self.results) == len(self.setup.rounds)

    @property
    def current_round(self) -> RoundSetup:
        if self.finished:
            raise RuntimeError("game is over")
        return self.setup.rounds[len(self.results)]

    @property
    def posting_phase(self) -> int | None:
        """0-based posting phase now open this round, or None once posting is done."""
        self.current_round  # raises if the game is over
        done = len(self._posts)
        return done if done < self.posting_phases else None

    def posts_so_far(self) -> list[tuple[str, ...]]:
        """This round's completed posting phases, each a message per seat."""
        return list(self._posts)

    def submit_posts(self, texts: tuple[str, ...] | list[str]) -> None:
        """All mediums' messages for the open posting phase, at once, by seat."""
        if self.posting_phase is None:
            raise RuntimeError("posting is over for this round")
        texts = tuple(texts)
        if len(texts) != self.setup.n_mediums:
            raise ValueError("one message per medium")
        for t in texts:
            if not isinstance(t, str):
                raise TypeError("messages must be strings")
            if word_count(t) > MAX_MESSAGE_WORDS:
                raise ValueError(f"message over {MAX_MESSAGE_WORDS} words")
        self._posts.append(texts)

    def submit_guesses(self, guesses: tuple[int, ...] | list[int]) -> RoundResult:
        if self.posting_phase is not None:
            raise RuntimeError("guesses come after every posting phase")
        rnd = self.current_round
        guesses = tuple(guesses)
        if len(guesses) != self.setup.n_mediums:
            raise ValueError("one guess per medium")
        for g in guesses:
            validate_guess(g)
        result = RoundResult(
            index=rnd.index,
            secret=rnd.secret,
            posts=tuple(self._posts),
            guesses=guesses,
            points=score_round(rnd.secret, guesses),
        )
        self.results.append(result)
        self._posts = []
        return result

    def totals(self) -> tuple[int, ...]:
        n = self.setup.n_mediums
        return tuple(sum(r.points[i] for r in self.results) for i in range(n))

    def payoffs(self) -> tuple[int, ...]:
        return payoffs(self.totals(), self.allies)
