"""
Unique module runner for motion-coder / orchestrator.

  python scripts/modules/run_module.py --id motionfix_semantic_edit --args-json '{...}'
  python scripts/modules/run_module.py --id speed_edit --args-json '{...}' --out result.json

Args JSON keys depend on module id (see BACKEND handlers below).
Returns JSON on stdout (and optionally --out): {ok, module, artifacts, seconds, error}.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKENDS = REPO_ROOT / "scripts" / "backends"
SESSION = REPO_ROOT / "scripts" / "session"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import backend_meta  # noqa: E402


def _env_python(kind: str) -> str:
    if kind == "mld":
        return os.environ.get(
            "MLD_PYTHON",
            str(Path.home() / "miniconda3" / "envs" / "mld" / "python.exe"),
        )
    if kind == "motionfix":
        return os.environ.get(
            "MOTIONFIX_PYTHON",
            str(Path.home() / "miniconda3" / "envs" / "motionfix" / "python.exe"),
        )
    return sys.executable


def _run(cmd: List[str], *, cwd: Optional[Path] = None) -> None:
    print(">>>", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    proc = subprocess.run(cmd, cwd=str(cwd or REPO_ROOT), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"exit {proc.returncode}: {' '.join(cmd)}")


def _ok(module: str, artifacts: Dict[str, Any], t0: float, **extra: Any) -> Dict[str, Any]:
    out = {
        "ok": True,
        "module": module,
        "artifacts": artifacts,
        "seconds": round(time.time() - t0, 3),
        "error": None,
    }
    out.update(extra)
    return out


def _fail(module: str, err: str, t0: float) -> Dict[str, Any]:
    return {
        "ok": False,
        "module": module,
        "artifacts": {},
        "seconds": round(time.time() - t0, 3),
        "error": err,
    }


def run_mld_generate(args: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    text = args.get("text") or args.get("mld_text")
    out = Path(args["out"])
    if not text:
        return _fail("mld_generate", "text/mld_text required", t0)
    length = int(args.get("length") or 60)
    py = _env_python("mld")
    try:
        _run(
            [
                py,
                str(BACKENDS / "mld_generate.py"),
                "--text",
                str(text),
                "--length",
                str(length),
                "--out",
                str(out),
            ]
        )
    except Exception as exc:  # noqa: BLE001
        return _fail("mld_generate", str(exc), t0)
    return _ok("mld_generate", {"joints_npy": str(out)}, t0)


def run_mld_bridge(args: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    joints = Path(args["joints"])
    out = Path(args["out"])
    py_mld = _env_python("mld")
    py_mf = _env_python("motionfix")
    iters = int(args.get("iters") or 40)
    target_len = int(args.get("target_len") or 0)
    try:
        cmd = [
            py_mld,
            str(BACKENDS / "mld_to_motionfix.py"),
            "--joints",
            str(joints),
            "--out",
            str(out),
            "--iters",
            str(iters),
            "--target-len",
            str(target_len),
        ]
        if args.get("max_frames"):
            cmd += ["--max-frames", str(args["max_frames"])]
        _run(cmd)
        _run(
            [
                py_mf,
                str(BACKENDS / "motionfix_facefront.py"),
                "--in",
                str(out),
                "--out",
                str(out),
            ]
        )
    except Exception as exc:  # noqa: BLE001
        return _fail("mld_bridge", str(exc), t0)
    return _ok("mld_bridge", {"source_npy": str(out)}, t0)


def run_motionfix_edit(args: Dict[str, Any], *, module: str = "motionfix_semantic_edit") -> Dict[str, Any]:
    t0 = time.time()
    out = Path(args["out"])
    py = _env_python("motionfix")
    cmd = [
        py,
        str(BACKENDS / "motionfix_edit.py"),
        "--out",
        str(out),
        "--steps",
        str(int(args.get("steps") or 0)),
        "--gd-text",
        str(float(args.get("gd_text") or 2.5)),
        "--init-from",
        str(args.get("init_from") or "source"),
        "--edit-mode",
        str(args.get("edit_mode") or "auto"),
        "--body-amp",
        str(float(args.get("body_amp") or 1.0)),
        "--no-adapt-bridged",
    ]
    if args.get("intent_json"):
        cmd += ["--intent-json", str(args["intent_json"])]
    elif args.get("text"):
        cmd += ["--text", str(args["text"])]
    if args.get("gd_motion") is not None:
        cmd += ["--gd-motion", str(args["gd_motion"])]
    if args.get("sdedit_strength") is not None:
        cmd += ["--sdedit-strength", str(args["sdedit_strength"])]
    if args.get("speed_scale") is not None:
        cmd += ["--speed-scale", str(args["speed_scale"])]
        if args.get("edit_mode") in (None, "auto"):
            # force geometric speed path
            idx = cmd.index("--edit-mode")
            cmd[idx + 1] = "speed"
    if args.get("source_id"):
        cmd += ["--source-id", str(args["source_id"])]
    elif args.get("source_npy"):
        cmd += ["--source-npy", str(args["source_npy"])]
    else:
        return _fail(module, "source_id or source_npy required", t0)
    try:
        _run(cmd)
    except Exception as exc:  # noqa: BLE001
        return _fail(module, str(exc), t0)
    return _ok(module, {"edited_npy": str(out)}, t0)


def run_speed_edit(args: Dict[str, Any]) -> Dict[str, Any]:
    a = dict(args)
    a["edit_mode"] = "speed"
    if a.get("speed_scale") is None and a.get("intent_json"):
        # motionfix_edit reads scale from intent constraints
        pass
    return run_motionfix_edit(a, module="speed_edit")


def run_quality_gate(args: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    edited = Path(args["edited"])
    out = Path(args["out"])
    py = _env_python("motionfix")
    cmd = [
        py,
        str(SESSION / "quality_gate.py"),
        "--edited",
        str(edited),
        "--out",
        str(out),
    ]
    if args.get("action"):
        cmd += ["--action", str(args["action"])]
    if args.get("intent_json"):
        cmd += ["--intent-json", str(args["intent_json"])]
    try:
        _run(cmd)
    except Exception as exc:  # noqa: BLE001
        return _fail("quality_gate", str(exc), t0)
    report: Dict[str, Any] = {}
    if out.is_file():
        report = json.loads(out.read_text(encoding="utf-8-sig"))
    return _ok(
        "quality_gate",
        {"quality_json": str(out)},
        t0,
        report=report,
    )


def run_sit_hold_postprocess(args: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    inp = Path(args["input"] if "input" in args else args["in"])
    out = Path(args["out"])
    py = _env_python("motionfix")
    cmd = [
        py,
        str(BACKENDS / "sit_hold_postprocess.py"),
        "--in",
        str(inp),
        "--out",
        str(out),
    ]
    if args.get("forbid"):
        forbid = args["forbid"]
        if isinstance(forbid, list):
            forbid = ",".join(forbid)
        cmd += ["--forbid", str(forbid)]
    if args.get("intent_json"):
        cmd += ["--intent-json", str(args["intent_json"])]
    try:
        _run(cmd)
    except Exception as exc:  # noqa: BLE001
        return _fail("sit_hold_postprocess", str(exc), t0)
    return _ok("sit_hold_postprocess", {"edited_npy": str(out)}, t0)


def run_timeline_postprocess(args: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    inp = Path(args["input"] if "input" in args else args["in"])
    out = Path(args["out"])
    py = _env_python("motionfix")
    cmd = [
        py,
        str(BACKENDS / "timeline_postprocess.py"),
        "--in",
        str(inp),
        "--out",
        str(out),
    ]
    if args.get("action"):
        cmd += ["--action", str(args["action"])]
    if args.get("intent_json"):
        cmd += ["--intent-json", str(args["intent_json"])]
    if args.get("timeline_json"):
        cmd += ["--timeline-json", str(args["timeline_json"])]
    try:
        _run(cmd)
    except Exception as exc:  # noqa: BLE001
        return _fail("timeline_postprocess", str(exc), t0)
    return _ok("timeline_postprocess", {"edited_npy": str(out)}, t0)


def run_wave_walk_postprocess(args: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    inp = Path(args["input"] if "input" in args else args["in"])
    out = Path(args["out"])
    py = _env_python("motionfix")
    cmd = [
        py,
        str(BACKENDS / "wave_walk_postprocess.py"),
        "--in",
        str(inp),
        "--out",
        str(out),
    ]
    if args.get("side"):
        cmd += ["--side", str(args["side"])]
    if args.get("min_forward") is not None:
        cmd += ["--min-forward", str(args["min_forward"])]
    if args.get("intent_json"):
        cmd += ["--intent-json", str(args["intent_json"])]
    try:
        _run(cmd)
    except Exception as exc:  # noqa: BLE001
        return _fail("wave_walk_postprocess", str(exc), t0)
    return _ok("wave_walk_postprocess", {"edited_npy": str(out)}, t0)


def run_visualize(args: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    inp = Path(args["input"])
    out = Path(args["out"])
    # Prefer motionfix env; MLD env also works for joints-only.
    py = _env_python(str(args.get("env") or "motionfix"))
    cmd = [
        py,
        str(BACKENDS / "visualize_motion.py"),
        "--input",
        str(inp),
        "--out",
        str(out),
        "--title",
        str(args.get("title") or "preview"),
    ]
    try:
        _run(cmd)
    except Exception as exc:  # noqa: BLE001
        return _fail("visualize", str(exc), t0)
    return _ok("visualize", {"preview_gif": str(out)}, t0)


def run_motion_caption(args: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    edited = Path(args["edited"] if "edited" in args else args["input"])
    out = Path(args["out"])
    cmd = [
        sys.executable,
        str(SESSION / "motion_caption.py"),
        "--edited",
        str(edited),
        "--out",
        str(out),
    ]
    if args.get("quality"):
        cmd += ["--quality", str(args["quality"])]
    if args.get("timeline"):
        cmd += ["--timeline", str(args["timeline"])]
    if args.get("action"):
        cmd += ["--action", str(args["action"])]
    if args.get("intent_text"):
        cmd += ["--intent-text", str(args["intent_text"])]
    try:
        _run(cmd)
    except Exception as exc:  # noqa: BLE001
        return _fail("motion_caption", str(exc), t0)
    return _ok("motion_caption", {"motion_caption": str(out)}, t0)


def run_simmotion_edit(args: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    source = Path(args.get("source_npy") or args.get("input") or args.get("in"))
    out = Path(args["out"] if "out" in args else args["out_npy"])
    text = str(args.get("text") or args.get("mf_text") or args.get("edit_text") or "")
    cmd = [
        sys.executable,
        str(BACKENDS / "simmotion_edit.py"),
        "--source-npy",
        str(source),
        "--text",
        text,
        "--out-npy",
        str(out),
    ]
    if args.get("allow_copy_fallback"):
        cmd.append("--allow-copy-fallback")
    try:
        _run(cmd)
    except Exception as exc:  # noqa: BLE001
        return _fail("simmotion_edit", str(exc), t0)
    return _ok("simmotion_edit", {"edited_npy": str(out)}, t0)


def run_mock(args: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    return _ok("mock", {"note": args.get("note") or "noop"}, t0)


HANDLERS = {
    "mld_generate": run_mld_generate,
    "mld_bridge": run_mld_bridge,
    "motionfix_semantic_edit": run_motionfix_edit,
    "speed_edit": run_speed_edit,
    "sit_hold_postprocess": run_sit_hold_postprocess,
    "timeline_postprocess": run_timeline_postprocess,
    "wave_walk_postprocess": run_wave_walk_postprocess,
    "quality_gate": run_quality_gate,
    "motion_caption": run_motion_caption,
    "simmotion_edit": run_simmotion_edit,
    "visualize": run_visualize,
    "mock": run_mock,
}


def run_module(module_id: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    args = args or {}
    meta = backend_meta(module_id)
    handler = HANDLERS.get(module_id)
    if handler is None:
        return {
            "ok": False,
            "module": module_id,
            "artifacts": {},
            "seconds": 0.0,
            "error": f"unknown module id: {module_id}",
            "meta": meta,
        }
    result = handler(args)
    result["meta"] = {
        "id": meta.get("id"),
        "name": meta.get("name"),
        "owner_agent": meta.get("owner_agent"),
        "mock": meta.get("mock"),
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one backend module by id")
    ap.add_argument("--id", required=True, help="Backend module id")
    ap.add_argument("--args-json", type=str, default="{}", help="JSON object of args")
    ap.add_argument("--args-file", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="Write result JSON here")
    ns = ap.parse_args()

    if ns.args_file:
        args = json.loads(ns.args_file.read_text(encoding="utf-8-sig"))
    else:
        args = json.loads(ns.args_json)

    result = run_module(ns.id, args)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if ns.out:
        ns.out.parent.mkdir(parents=True, exist_ok=True)
        ns.out.write_text(text, encoding="utf-8")
    print(text)
    if not result.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
