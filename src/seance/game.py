"""Pure, seeded rules engine for the Séance game.

No I/O and no model calls. Given a seed, `deal` fixes every secret, vision and
speaking order for the whole game up front, so a game can be replayed exactly.
`Game` then enforces the order of play: each round, every medium speaks once
in the round's speaking order, then all guesses arrive together, then the
round is scored and revealed.

Rules (see the sprint note "Séance game: final design (2026-10-23)"):

- 4 mediums, 10 rounds.
- Each round the secret is a whole number from 1 to 100, uniform.
- Each medium privately gets a vision: 40 consecutive integers containing the
  secret, with the secret at a uniformly random position (offset 0..39) inside
  it. Visions are not clipped to 1..100, since clipping would leak where the
  secret is.
- Speaking order is round-robin and rotates: in round r (0-based) the first
  speaker is medium r mod 4, then the others in seat order.
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
    speaking_order: tuple[int, ...]  # seats


@dataclass(frozen=True)
class GameSetup:
    seed: int
    rounds: tuple[RoundSetup, ...]
    n_mediums: int = N_MEDIUMS


def speaking_order(round_index: int, n_mediums: int = N_MEDIUMS) -> tuple[int, ...]:
    first = round_index % n_mediums
    return tuple((first + k) % n_mediums for k in range(n_mediums))


def deal_vision(secret: int, rng: random.Random) -> Vision:
    offset = rng.randrange(VISION_WIDTH)  # position of the secret inside the range
    lo = secret - offset
    return Vision(lo, lo + VISION_WIDTH - 1)


def deal(seed: int, n_rounds: int = N_ROUNDS, n_mediums: int = N_MEDIUMS) -> GameSetup:
    """Fix every secret, vision and speaking order for a game from its seed."""
    rng = random.Random(seed)
    rounds = []
    for r in range(n_rounds):
        secret = rng.randint(SECRET_MIN, SECRET_MAX)
        visions = tuple(deal_vision(secret, rng) for _ in range(n_mediums))
        rounds.append(RoundSetup(r, secret, visions, speaking_order(r, n_mediums)))
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
    messages: list[tuple[int, str]]  # (seat, text) in speaking order
    guesses: tuple[int, ...]
    points: tuple[int, ...]


@dataclass
class Game:
    """Enforces the order of play over a dealt GameSetup."""

    setup: GameSetup
    allies: tuple[int, int] | None = None
    results: list[RoundResult] = field(default_factory=list)
    _messages: list[tuple[int, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
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

    def next_speaker(self) -> int | None:
        """Seat whose turn it is to speak, or None once everyone has spoken."""
        order = self.current_round.speaking_order
        return order[len(self._messages)] if len(self._messages) < len(order) else None

    def messages_so_far(self) -> list[tuple[int, str]]:
        return list(self._messages)

    def submit_message(self, seat: int, text: str) -> None:
        expected = self.next_speaker()
        if expected is None:
            raise RuntimeError("everyone has already spoken this round")
        if seat != expected:
            raise RuntimeError(f"seat {seat} spoke out of turn; expected {expected}")
        if word_count(text) > MAX_MESSAGE_WORDS:
            raise ValueError(f"message over {MAX_MESSAGE_WORDS} words")
        self._messages.append((seat, text))

    def submit_guesses(self, guesses: tuple[int, ...] | list[int]) -> RoundResult:
        if self.next_speaker() is not None:
            raise RuntimeError("guesses come after everyone has spoken")
        rnd = self.current_round
        guesses = tuple(guesses)
        if len(guesses) != self.setup.n_mediums:
            raise ValueError("one guess per medium")
        for g in guesses:
            validate_guess(g)
        result = RoundResult(
            index=rnd.index,
            secret=rnd.secret,
            messages=list(self._messages),
            guesses=guesses,
            points=score_round(rnd.secret, guesses),
        )
        self.results.append(result)
        self._messages = []
        return result

    def totals(self) -> tuple[int, ...]:
        n = self.setup.n_mediums
        return tuple(sum(r.points[i] for r in self.results) for i in range(n))

    def payoffs(self) -> tuple[int, ...]:
        return payoffs(self.totals(), self.allies)
