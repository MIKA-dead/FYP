"""Open-vocabulary edit routing (SimMotionEdit) — plumbing, without the model.

The real backend needs a clone + checkpoints that are not in this repo, so this
test stands a fake one up: a directory that *looks* ready and a command that
copies the source clip. What is under test is the routing decision and the
after-edit measurement, not the generative quality.

  python scripts/tests/test_open_vocab.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in ("agent", "backends", "session", "orchestrator"):
    sys.path.insert(0, str(REPO / "scripts" / _p))

CHECKS: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def _fake_backend(tmp: Path) -> Path:
    root = tmp / "SimMotionEdit"
    (root / "checkpoints").mkdir(parents=True)
    (root / "checkpoints" / "weights.bin").write_bytes(b"x")
    (root / "README.md").write_text("fake", encoding="utf-8")
    (root / "sample.py").write_text("# fake entry\n", encoding="utf-8")
    return root


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="open_vocab_test_"))
    import edits

    # 1) default: closed set only, unknown wording is rejected.
    for var in ("AGENT_OPEN_VOCAB_EDIT", "SIMMOTION_ROOT", "SIMMOTION_CMD_JSON"):
        os.environ.pop(var, None)
    check("closed_set_by_default", edits.open_vocab_available(refresh=True) is False,
          "AGENT_OPEN_VOCAB_EDIT unset")
    default_cls = edits.classify("sway the hips gently")
    check("unknown_wording_rejected_by_default",
          default_cls["supported"] is False and not default_cls.get("open_vocab"),
          str(default_cls.get("suggestion")))

    # 2) with a ready backend, the same wording becomes routable.
    root = _fake_backend(tmp)
    os.environ["SIMMOTION_ROOT"] = str(root)
    os.environ["AGENT_OPEN_VOCAB_EDIT"] = "1"
    os.environ["SIMMOTION_CMD_JSON"] = json.dumps([
        sys.executable, "-c",
        "import shutil,sys; shutil.copy2(r'{source}', r'{out}')",
    ])
    import simmotion_edit

    disc = simmotion_edit.discover()
    check("fake_backend_ready", bool(disc.get("ready")), str(disc.get("entry")))
    check("open_vocab_enabled", edits.open_vocab_available(refresh=True) is True)
    cls = edits.classify("sway the hips gently")
    check("unknown_wording_routable", cls.get("open_vocab") is True, str(cls.get("family")))
    props = edits.classify("pick up the cup")
    check("props_stay_impossible", not props.get("open_vocab"), str(props.get("family")))
    check("capability_text_mentions_backend",
          "open-vocabulary" in edits.capability_text(), "")

    # 3) the tool actually routes, and still measures the result.
    from agent import run_agent  # noqa: E402  (import check only)

    _ = run_agent
    from tools import MotionTools  # noqa: E402

    tools = MotionTools(tmp / "ep", generator="dataset", skip_viz=True,
                        goal="a person walks forward")
    obs = tools.generate_motion("a person walks forward")
    check("have_a_clip", bool(obs.get("ok")), str(obs.get("error") or ""))
    edit = tools.edit_motion("sway the hips gently")
    check("routed_to_open_vocab", edit.get("backend") == "simmotion",
          f"backend={edit.get('backend')} error={edit.get('error')}")

    # 4) a backend that produces no clip must fail, not silently "succeed".
    os.environ["SIMMOTION_CMD_JSON"] = json.dumps([sys.executable, "-c", "raise SystemExit(3)"])
    tools2 = MotionTools(tmp / "ep2", generator="dataset", skip_viz=True,
                         goal="a person walks forward")
    tools2.generate_motion("a person walks forward")
    failed = tools2.edit_motion("sway the hips gently")
    check("failed_backend_reported", failed.get("ok") is False
          and failed.get("error"), f"{failed.get('error')}")

    report = {"checks": CHECKS, "passed": sum(c["ok"] for c in CHECKS), "n": len(CHECKS)}
    (tmp / "open_vocab_checks.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n{report['passed']}/{report['n']} checks passed — {tmp / 'open_vocab_checks.json'}")
    return 0 if report["passed"] == report["n"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

