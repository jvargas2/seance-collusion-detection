"""Number redaction for public messages (exploratory, `GameSpec.redact`).

Every number in a public message, in digits or in words, becomes
`[redacted]` before any player or the record's public view sees it. The
original text is kept in the record, privately, for analysis.

What counts as a number:

- digit runs, with a leading minus, inner decimal points, commas, colons or
  slashes ("-3", "4.5", "1,000", "12:05", "1/2") and an attached ordinal or
  plural ending ("21st", "40s", "40ish"); any Unicode decimal digit, and
  superscript and subscript digits;
- number words, any case: zero to nineteen, the tens (twenty to ninety, and
  the common misspelling "fourty"), hundred, thousand, million, dozen; their
  ordinals (first, second, ... ninetieth, hundredth); plurals ("tens",
  "twenties", "dozens"), "teens" and "-ish" forms ("fortyish"). A run of
  number words joined by spaces, hyphens or "and" ("forty-seven", "forty
  seven", "one hundred and five") becomes a single `[redacted]`.

"one" is redacted in every use ("no one" becomes "no [redacted]"), because
players are told every number word goes. Deliberately not redacted, since
they are mostly ordinary words here: "score", "half", "couple", "pair",
"single" (so "single digits" survives), "double", "quarter" (used as "upper
quarter" of a range), "once", "twice", Roman numerals and number words in
other languages. `tests/test_redact.py` pins both lists.
"""

from __future__ import annotations

import re

REDACTED = "[redacted]"

_UNITS = "zero one two three four five six seven eight nine".split()
_TEENS = "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "twenty thirty forty fourty fifty sixty seventy eighty ninety".split()
_BIG = "hundred thousand million dozen".split()
_ORDINALS = (
    "zeroth first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth "
    "thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth "
    "twentieth thirtieth fortieth fourtieth fiftieth sixtieth seventieth eightieth ninetieth "
    "hundredth thousandth millionth"
).split()


def _forms() -> list[str]:
    words = set(_ORDINALS) | {w + "s" for w in _ORDINALS}
    for w in _UNITS + _TEENS + _BIG:
        words |= {w, w + "s", w + "ish"}
    for w in _TENS:  # twenty -> twenties, twentyish
        words |= {w, w[:-1] + "ies", w + "ish"}
    words |= {"sixes", "nought", "naught", "teens"}  # "the low teens" names a decade, like "twenties"
    return sorted(words, key=len, reverse=True)  # longest first, so "sixteen" beats "six"


_WORD = r"(?:" + "|".join(map(re.escape, _forms())) + r")"
_WORDS = re.compile(rf"(?<![^\W\d_]){_WORD}(?:(?:[\s-]+|\s+and\s+){_WORD})*(?![^\W\d_])", re.IGNORECASE)
_SUP_SUB = "⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉"
_D = rf"[\d{_SUP_SUB}]"
_DIGITS = re.compile(
    rf"(?:(?<!\w)-)?{_D}+(?:[.,:/]{_D}+)*(?:(?:st|nd|rd|th|s|ish)(?![^\W\d_]))?", re.IGNORECASE
)


def redact(text: str) -> str:
    """`text` with every number replaced by `[redacted]`."""
    return _WORDS.sub(REDACTED, _DIGITS.sub(REDACTED, text))


def has_number(text: str) -> bool:
    return redact(text) != text
