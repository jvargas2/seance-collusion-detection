"""Prompt builders and the two model calls each medium makes per round.

speak -> {"private_note": str, "public_message": str}
guess -> {"private_note": str, "guess": int}

Every reply is validated; an invalid one is sent back with the error and
retried. Every attempt's raw text is logged so nothing a model said is lost.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .conditions import ROLE_ALLY, ROLE_ALLY_GIVEN_CODE, ROLE_SOLO
from .game import (
    DEFAULT_POSTING_PHASES,
    MAX_MESSAGE_WORDS,
    MEDIUM_NAMES,
    N_ROUNDS,
    SECRET_MAX,
    SECRET_MIN,
    VISION_WIDTH,
    Game,
    word_count,
)
from .inference import ChatClient
from .skins import Skin
from .skins import templates as T

SPEAK = "speak"
GUESS = "guess"

SPEAK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "private_note": {"type": "string"},
        "public_message": {"type": "string"},
    },
    "required": ["private_note", "public_message"],
    "additionalProperties": False,
}

GUESS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "private_note": {"type": "string"},
        "guess": {"type": "integer", "minimum": SECRET_MIN, "maximum": SECRET_MAX},
    },
    "required": ["private_note", "guess"],
    "additionalProperties": False,
}

SCHEMAS = {SPEAK: SPEAK_SCHEMA, GUESS: GUESS_SCHEMA}


# --- prompts ----------------------------------------------------------------


def _join_names(names: list[str]) -> str:
    return ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0]


def system_prompt(
    skin: Skin, seat: int, role: str, partner: int | None, posting_phases: int = DEFAULT_POSTING_PHASES
) -> str:
    common = dict(
        n_mediums=len(MEDIUM_NAMES),
        n_rounds=N_ROUNDS,
        secret_min=SECRET_MIN,
        secret_max=SECRET_MAX,
        width=VISION_WIDTH,
        max_words=MAX_MESSAGE_WORDS,
    )
    posting = {1: T.POSTING_RULE_ONE, 2: T.POSTING_RULE_TWO}[posting_phases]
    posting_rule = skin.render(posting, n_mediums=common["n_mediums"], max_words=common["max_words"])
    parts = [
        skin.render(
            T.RULES,
            me=MEDIUM_NAMES[seat],
            names=_join_names(list(MEDIUM_NAMES)),
            posting_rule=posting_rule,
            **common,
        )
    ]
    if role == ROLE_SOLO:
        if partner is not None:
            raise ValueError("solo mediums have no partner")
        parts.append(skin.render(T.GOAL_SOLO, n_rounds=common["n_rounds"]))
    elif role in (ROLE_ALLY, ROLE_ALLY_GIVEN_CODE):
        if partner is None:
            raise ValueError("allies need a partner")
        p = MEDIUM_NAMES[partner]
        parts.append(skin.render(T.GOAL_ALLY, partner=p))
        code = T.CODE_IMPROVISE if role == ROLE_ALLY else T.CODE_GIVEN
        parts.append(skin.render(code, partner=p))
    else:
        raise ValueError(f"unknown role {role!r}")
    return "\n\n".join(parts)


def _num(v: float) -> str:
    """3.0 -> "3", 2.5 -> "2.5"."""
    return f"{v:g}"


def _seq(values: tuple[float, ...] | list[float]) -> str:
    return ", ".join(f"{MEDIUM_NAMES[i]} {_num(v)}" for i, v in enumerate(values))


def _quote(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def _posts_lines(posts: tuple[tuple[str, ...], ...] | list[tuple[str, ...]], n_phases: int) -> list[str]:
    lines = []
    for k, texts in enumerate(posts):
        if n_phases == 1:
            lines.append(T.MESSAGES_LINE)
        else:
            lines.append(T.MESSAGES_PHASE_LINE.format(phase=k + 1, n_phases=n_phases))
        for speaker, text in enumerate(texts):
            lines.append(T.MESSAGE_ITEM.format(speaker=MEDIUM_NAMES[speaker], text=_quote(text)))
    return lines


def _speak_notes(round_notes: dict) -> str:
    return T.NOTE_JOIN.join(_quote(n) for n in round_notes.get(SPEAK, []))


def user_prompt(
    skin: Skin,
    game: Game,
    seat: int,
    phase: str,
    notes: list[dict],
    partner: int | None,
) -> str:
    """Everything `seat` may know right now: the public record so far, its own
    visions and its own earlier private notes. Nothing about anyone else's.

    `notes[r]` holds the seat's notes for round r (0-based): {"speak": [one
    note per posting phase], "guess": note}. The current round's entry holds
    whatever this seat has written so far this round."""
    n_rounds = len(game.setup.rounds)
    n_phases = game.posting_phases
    lines: list[str] = []
    if not game.results:
        lines.append(T.NO_HISTORY)
    for res in game.results:
        v = game.setup.rounds[res.index].visions[seat]
        lines.append(T.ROUND_HEADER_PAST.format(round=res.index + 1, n_rounds=n_rounds))
        lines.append(skin.render(T.YOUR_CLUE_PAST, lo=v.lo, hi=v.hi, mid=v.middle))
        lines.extend(_posts_lines(res.posts, n_phases))
        lines.append(T.GUESSES_LINE.format(guesses=_seq(res.guesses)))
        lines.append(T.REVEAL_LINE.format(secret=res.secret, points=_seq(res.points)))
        if res.index < len(notes):
            n = notes[res.index]
            lines.append(T.NOTES_LINE.format(speak_note=_speak_notes(n), guess_note=_quote(n.get(GUESS, ""))))
    if game.results:
        totals = game.totals()
        lines.append(T.TOTALS_LINE.format(totals=_seq(totals)))
        if partner is not None:
            lines.append(
                T.POOLED_LINE.format(partner=MEDIUM_NAMES[partner], pooled=_num(totals[seat] + totals[partner]))
            )

    rnd = game.current_round
    v = rnd.visions[seat]
    lines.append("")
    lines.append(T.ROUND_HEADER_NOW.format(round=rnd.index + 1, n_rounds=n_rounds))
    lines.append(skin.render(T.YOUR_CLUE_NOW, lo=v.lo, hi=v.hi, mid=v.middle))
    so_far = game.posts_so_far()
    if so_far:
        lines.extend(_posts_lines(so_far, n_phases))
    else:
        lines.append(T.NO_MESSAGES_YET)
    if rnd.index < len(notes) and notes[rnd.index].get(SPEAK):
        lines.append(T.CURRENT_NOTE_LINE.format(speak_note=_speak_notes(notes[rnd.index])))
    lines.append("")
    if phase == SPEAK:
        if game.posting_phase is None:
            raise RuntimeError("posting is over for this round")
        if n_phases > 1:
            lines.append(T.ASK_SPEAK_PHASE.format(phase=game.posting_phase + 1, n_phases=n_phases))
        lines.append(skin.render(T.ASK_SPEAK, max_words=MAX_MESSAGE_WORDS))
    elif phase == GUESS:
        if game.posting_phase is not None:
            raise RuntimeError("guesses come after every posting phase")
        lines.append(skin.render(T.ASK_GUESS, secret_min=SECRET_MIN, secret_max=SECRET_MAX))
    else:
        raise ValueError(f"unknown phase {phase!r}")
    return "\n".join(lines)


# --- parsing and validation -------------------------------------------------


class ReplyError(ValueError):
    """A reply that can't be used. `too_long_only` marks a message whose only
    problem is length, which the last attempt may truncate instead of failing."""

    def __init__(self, msg: str, too_long_only: bool = False, parsed: dict | None = None):
        super().__init__(msg)
        self.too_long_only = too_long_only
        self.parsed = parsed


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def extract_json(text: str) -> dict:
    s = _THINK.sub("", text).strip()
    if "</think>" in s:  # reasoning without an opening tag
        s = s.split("</think>", 1)[1].strip()
    m = _FENCE.match(s)
    if m:
        s = m.group(1)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("{"), s.rfind("}")
        if start == -1 or end <= start:
            raise ReplyError("The reply was not a JSON object.") from None
        try:
            obj = json.loads(s[start : end + 1])
        except json.JSONDecodeError as e:
            raise ReplyError(f"The reply was not valid JSON ({e.msg}).") from None
    if not isinstance(obj, dict):
        raise ReplyError("The reply was not a JSON object.")
    return obj


def _note(obj: dict) -> str:
    note = obj.get("private_note")
    if not isinstance(note, str):
        raise ReplyError('"private_note" must be a string.')
    return note


def parse_speak(text: str) -> dict:
    obj = extract_json(text)
    note = _note(obj)
    msg = obj.get("public_message")
    if not isinstance(msg, str) or not msg.strip():
        raise ReplyError('"public_message" must be a non-empty string.')
    msg = " ".join(msg.split())  # one line, single spaces
    parsed = {"private_note": note, "public_message": msg}
    n = word_count(msg)
    if n > MAX_MESSAGE_WORDS:
        raise ReplyError(
            f'"public_message" has {n} words; the limit is {MAX_MESSAGE_WORDS}.',
            too_long_only=True,
            parsed=parsed,
        )
    return parsed


def parse_guess(text: str) -> dict:
    obj = extract_json(text)
    note = _note(obj)
    g = obj.get("guess")
    if isinstance(g, bool):
        g = None
    elif isinstance(g, float) and g.is_integer():
        g = int(g)
    elif isinstance(g, str) and re.fullmatch(r"\s*-?\d+\s*", g):
        g = int(g)
    if not isinstance(g, int):
        raise ReplyError('"guess" must be a whole number.')
    if not SECRET_MIN <= g <= SECRET_MAX:
        raise ReplyError(f'"guess" must be from {SECRET_MIN} to {SECRET_MAX}; got {g}.')
    return {"private_note": note, "guess": g}


PARSERS = {SPEAK: parse_speak, GUESS: parse_guess}


def truncate_words(text: str, limit: int = MAX_MESSAGE_WORDS) -> str:
    return " ".join(text.split()[:limit])


# --- calls ------------------------------------------------------------------


class AgentCallFailed(RuntimeError):
    def __init__(self, msg: str, log: CallLog):
        super().__init__(msg)
        self.log = log


@dataclass
class CallLog:
    round: int
    seat: int
    phase: str
    user_prompt: str
    attempts: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    truncated: bool = False
    posting_phase: int | None = None  # 0-based, speak calls only

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round + 1,
            "medium": MEDIUM_NAMES[self.seat],
            "phase": self.phase,
            "posting_phase": None if self.posting_phase is None else self.posting_phase + 1,
            "user_prompt": self.user_prompt,
            "attempts": self.attempts,
            "result": self.result,
            "truncated": self.truncated,
        }

    @property
    def prompt_tokens(self) -> int:
        return sum(a.get("prompt_tokens", 0) for a in self.attempts)

    @property
    def completion_tokens(self) -> int:
        return sum(a.get("completion_tokens", 0) for a in self.attempts)


async def call_medium(
    client: ChatClient,
    skin: Skin,
    system: str,
    game: Game,
    seat: int,
    phase: str,
    notes: list[dict],
    partner: int | None,
    max_attempts: int = 3,
) -> tuple[dict, CallLog]:
    """One validated call. Raises AgentCallFailed after max_attempts bad replies."""
    user = user_prompt(skin, game, seat, phase, notes, partner)
    log = CallLog(
        round=game.current_round.index,
        seat=seat,
        phase=phase,
        user_prompt=user,
        posting_phase=game.posting_phase if phase == SPEAK else None,
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    parse = PARSERS[phase]
    for attempt in range(1, max_attempts + 1):
        comp = await client.complete(messages, f"seance_{phase}", SCHEMAS[phase])
        entry: dict[str, Any] = {
            "raw": comp.text,
            "model": comp.model,
            "prompt_tokens": comp.prompt_tokens,
            "completion_tokens": comp.completion_tokens,
            "latency_s": round(comp.latency_s, 3),
            "finish_reason": comp.finish_reason,
            "reasoning_chars": comp.reasoning_chars,
            "error": None,
        }
        log.attempts.append(entry)
        try:
            log.result = parse(comp.text)
            return log.result, log
        except ReplyError as e:
            entry["error"] = str(e)
            if attempt == max_attempts and e.too_long_only and e.parsed is not None:
                result = dict(e.parsed)
                result["public_message"] = truncate_words(result["public_message"])
                log.result, log.truncated = result, True
                return result, log
            messages = messages + [
                {"role": "assistant", "content": comp.text},
                {"role": "user", "content": skin.render(T.RETRY, error=str(e))},
            ]
    raise AgentCallFailed(
        f"{MEDIUM_NAMES[seat]} gave no usable {phase} reply in {max_attempts} attempts", log
    )
