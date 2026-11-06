"""Which free-form edit instructions the backend can actually apply.

Before this module the capability list was a paragraph of prose in the system
prompt, and the only way to learn that an instruction was impossible was to
pay for the edit (≈15 s), watch the clip come back byte-identical, and read
``no_change: true``. That is a poor contract for an agent: it burns a step and
learns nothing reusable.

Here the list is executable, single-source, and used three ways:

* ``capability_text()`` — the text the controller sees (prompt and tool schema);
* ``classify()`` — gates ``edit_motion`` before any backend work, so an
  impossible instruction returns ``unsupported_instruction`` with a suggestion
  in milliseconds instead of a wasted diffusion pass;
* ``benchmark`` — an honest ``instruction_coverage`` metric: of the
  instructions the agent actually wrote, how many were inside the tool's
  ability at all.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Family:
    name: str
    pattern: str
    canonical: str
    note: str = ""
    params: Dict[str, str] = field(default_factory=dict)
    suggestion: str = ""
    _rx: Optional[re.Pattern] = field(default=None, repr=False)

    def match(self, text: str) -> Optional[re.Match]:
        if self._rx is None:
            self._rx = re.compile(self.pattern, re.I)
        return self._rx.search(text)


# Families are ordered: the first match wins, so put specific before general.
SUPPORTED: Tuple[Family, ...] = (
    Family(
        "stand_up",
        r"\bstand(\s+up|ing)?\b(?!.*\barm\b)",
        "stand up from the sit",
        "Removes the seated hold (reverse of sit_down).",
    ),
    Family(
        "sit_remove_preamble",
        r"\b(no|remove|without|skip)\b.*\bwalk(ing)?\b.*\bsit\b|\bsit\b.*\b(immediately|in place)\b",
        "sit down immediately, no walking before sitting",
        "Trims the walking preamble of a sit clip.",
    ),
    Family(
        "sit_deeper",
        r"\b(sit|sitting|seat)\b.*\b(lower|deeper|more)\b|\b(lower|deeper)\b.*\b(sit|seat)\b",
        "sit down lower (deeper into the seat)",
        "Lowers the pelvis of the seated hold and re-plants the feet.",
    ),
    Family(
        "arm_raise",
        r"\b(raise|lift|put)\b.*\b(arm|hand|wrist)\b|\b(arm|hand)\b.*\b(higher|up|above)\b",
        "raise the right arm higher",
        "One arm at a time; the other arm is preserved.",
        {"side": "left|right", "level": "higher|high"},
    ),
    Family(
        "arm_lower",
        r"\b(lower|drop|put down|bring down)\b.*\b(arm|hand)\b",
        "lower the right arm",
        "One arm at a time.",
        {"side": "left|right"},
    ),
    Family(
        "wave",
        r"\bwave|waving|hand wave\b",
        "wave with the right hand",
        "One-arm wave; use arm_raise for a static raised arm.",
        {"side": "left|right"},
    ),
    Family(
        "speed",
        r"\b(faster|slower|slow down|speed up|quicker|speed)\b",
        "walk slower",
        "Tempo retime; the pose sequence is preserved.",
    ),
    Family(
        "walk",
        r"\b(walk|step|move)\b.*\b(forward|front|more)\b|\bwalk\b(?!.*\bstop\b)",
        "walk forward",
        "Forward locomotion; it can only scale travel the clip already has.",
        {"direction": "forward|in_place|backward"},
    ),
    Family(
        "turn",
        r"\bturn|rotate|spin|about face|180\b",
        "turn around in place",
        "Stops or limits a turn. A much larger NEW turn cannot be invented.",
        {"direction": "left|right|around", "amount": "90|180"},
    ),
    Family(
        "trim_trailing",
        r"\b(trim|cut|remove|shorten|stop)\b.*\b(end|trailing|after|extra|tail)\b|\btoo long\b",
        "remove the extra motion after sitting",
        "Cuts frames after the action completes.",
    ),
    Family(
        "wave_stop",
        r"\b(stop|no more|remove)\b.*\b(wave|waving)\b",
        "stop waving",
        "Lowers the waving arm.",
    ),
)

# Things people ask for that this backend cannot do at all. We name them so the
# rejection is informative instead of a generic failure.
UNSUPPORTED: Tuple[Family, ...] = (
    Family("object", r"\b(pick|hold|grab|throw|put on|wear|carry|lift)\b.*\b(object|ball|cup|box|bag|chair|door)\b",
           "add or manipulate an object", "The pipeline outputs motion only — no props.",
           suggestion="edit the body instead, e.g. 'raise the right arm higher'"),
    Family("camera", r"\bcamera|zoom|close[- ]up|angle\b", "change the camera",
           "Camera is not part of the clip.",
           suggestion="the clip has no camera to change; edit the motion instead"),
    Family("appearance", r"\b(wear|clothes|outfit|shirt|dress|hair|face|skin|color)\b",
           "change appearance", "Appearance/identity is frozen.",
           suggestion="appearance is frozen; edit the motion instead"),
    Family("body", r"\b(taller|shorter|fatter|thinner|muscle|body shape)\b", "change body shape",
           "Body shape is fixed by the skeleton.",
           suggestion="body shape is fixed; edit the motion instead"),
    Family("jump", r"\bjump|leap|hop\b", "jump", "No jump operator; re-generate instead.",
           suggestion="no jump operator — generate a new clip with that prompt"),
    Family("kick", r"\bkick|punch|hit\b", "kick or punch", "No impact operator.",
           suggestion="no impact operator — generate a new clip with that prompt"),
    Family("dance", r"\bdance|dancing|ballet|salsa\b", "dance", "Out of the editing distribution.",
           suggestion="generate a new clip with that prompt instead of editing"),
    Family("new_action", r"\b(fall|lie down|climb|swim|crawl|kneel)\b", "a new whole-body action",
           "Editing is relative; re-generate instead.",
           suggestion="generate a new clip with that prompt instead of editing"),
)

# Requests that *contradict* what the clip already does. A relative operator can
# only scale or extend what is there, so these are checked before the supported
# list: "walk in place" contains "walk" and would otherwise match it.
BLOCKERS: Tuple[Family, ...] = (
    Family(
        "in_place",
        r"\b(in[ -]place|on the spot|without moving|no forward travel|"
        r"keep the root planted|staying on the spot|stationary)\b",
        "walk in place",
        "The editor can scale or extend travel but cannot remove it.",
        suggestion="put 'walks in place' in the generation prompt; the editor "
                   "cannot remove forward travel",
    ),
    Family(
        "still",
        r"\b(freeze|hold still|stop moving|do not move)\b",
        "hold still",
        "Removing all motion is not an edit the backend can apply.",
        suggestion="generate a clip whose prompt already asks for the still pose",
    ),
)


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z]+", (text or "").lower()))


def _closest(text: str, pool) -> Optional[str]:
    """Suggestion for an impossible request: the family sharing the most words.

    Only the family name and canonical phrasing count; matching against the
    regex source would mostly hit syntax characters.
    """
    got = _tokens(text)
    if not got:
        return None
    best, best_n = None, 0
    for fam in pool:
        n = len(got & _tokens(fam.name.replace("_", " ") + " " + fam.canonical))
        if n > best_n:
            best, best_n = fam, n
    # One shared word ("up", "the") is noise, not a suggestion.
    return best.canonical if best and best_n >= 2 else None


def classify(instruction: str) -> Dict[str, Any]:
    """Decide whether ``instruction`` is inside the editor's ability.

    Returns a dict with ``supported``, ``family``, ``canonical`` (what to say
    instead), and ``reason``. Never raises, so callers can gate cheaply.
    """
    text = " ".join((instruction or "").lower().split())
    if not text:
        return {
            "supported": False,
            "family": None,
            "canonical": None,
            "reason": "empty instruction",
            "suggestion": "say one concrete change, e.g. 'raise the right arm higher'",
        }
    # Contradictions first: "walk in place" contains "walk", so the supported
    # walk family would otherwise claim it. "in place" is only a contradiction
    # for locomotion — "turn in place" is a rotation on the spot, which is
    # supported.
    rotating = re.search(r"\b(turn|spin|rotate|sit|stand|wave)\b", text) is not None
    for fam in BLOCKERS:
        if fam.name == "in_place" and rotating:
            continue
        if fam.match(text):
            return {
                "supported": False,
                "family": fam.name,
                "canonical": fam.canonical,
                "reason": f"{fam.canonical}: {fam.note}",
                "suggestion": fam.suggestion or _closest(text, SUPPORTED),
            }
    for fam in SUPPORTED:
        if fam.match(text):
            return {
                "supported": True,
                "family": fam.name,
                "canonical": fam.canonical,
                "reason": fam.note,
                "params": fam.params,
            }
    for fam in UNSUPPORTED:
        if fam.match(text):
            return {
                "supported": False,
                "family": fam.name,
                "canonical": None,
                "reason": f"{fam.canonical}: {fam.note}",
                "suggestion": fam.suggestion or _closest(text, SUPPORTED),
            }
    return {
        "supported": False,
        "family": None,
        "canonical": None,
        "reason": "no operator matches this instruction",
        "suggestion": _closest(text, SUPPORTED),
    }


def capability_text() -> str:
    """The capability paragraph shown to the controller."""
    lines = ["The editing backend applies ONE relative change per call. Supported:"]
    for fam in SUPPORTED:
        lines.append(f"- {fam.name}: \"{fam.canonical}\"" + (f" — {fam.note}" if fam.note else ""))
    lines.append("")
    lines.append("It cannot: " + "; ".join(f.canonical for f in UNSUPPORTED) + ".")
    lines.append("It also cannot undo what the clip already does: "
                 + "; ".join(f.canonical for f in BLOCKERS) + ".")
    lines.append("If the request needs one of those, finish and say it is out of scope — "
                 "do not spend an edit on it.")
    return "\n".join(lines)


def supported_names() -> List[str]:
    return [f.name for f in SUPPORTED]
