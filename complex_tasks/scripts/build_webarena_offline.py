#!/usr/bin/env python3
"""Build a text-only WebArena next-action dataset from AutoLibra traces.

The source mirror stores every observation/action as a timestamped JSON file.
This script pairs each agent observation with the immediately following action
and attaches the user task text. No browser or WebArena server is required.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


STAMP_RE = re.compile(r"_(\d{4}-\d\d-\d\dT[^.]+(?:\.\d+)?)\.json$")


def stamp(path: Path) -> str:
    match = STAMP_RE.search(path.name)
    return match.group(1) if match else path.name


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def task_text(instance: Path) -> str:
    files = sorted((instance / "user" / "json_data").glob("*user_action*.json"))
    for path in files:
        value = read_json(path)
        text = str(value.get("text") or value.get("intent") or "").strip()
        if text:
            return text
    return ""


def build_instance(instance: Path) -> list[dict[str, Any]]:
    task = task_text(instance)
    agent = instance / "agent" / "json_data"
    observations = sorted(agent.glob("*agent_observation*.json"), key=stamp)
    actions = sorted(agent.glob("*agent_action*.json"), key=stamp)
    if not task or not observations or not actions:
        return []

    events = [(stamp(p), "observation", p) for p in observations]
    events += [(stamp(p), "action", p) for p in actions]
    events.sort(key=lambda item: item[0])
    rows: list[dict[str, Any]] = []
    pending: tuple[str, Path] | None = None
    step = 0
    for ts, kind, path in events:
        if kind == "observation":
            pending = (ts, path)
            continue
        if pending is None:
            continue
        obs_ts, obs_path = pending
        obs = read_json(obs_path)
        action = read_json(path)
        html = str(obs.get("html") or obs.get("observation") or "").strip()
        if html and action.get("function"):
            rows.append(
                {
                    "sample_id": f"{instance.name}:{step}",
                    "trajectory_id": instance.name,
                    "step": step,
                    "task": task,
                    "url": str(obs.get("url") or ""),
                    "observation": html,
                    "reference_action": {
                        "function": action.get("function"),
                        "kwargs": action.get("kwargs") or {},
                    },
                    "reference_description": str(action.get("description") or ""),
                    "source_observation_timestamp": obs_ts,
                    "source_action_timestamp": ts,
                }
            )
            step += 1
        pending = None
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-trajectories", type=int, default=0)
    args = parser.parse_args()

    instances_root = args.source / "webarena" / "instances"
    if not instances_root.is_dir():
        raise SystemExit(f"Missing trace directory: {instances_root}")
    instances = sorted(p for p in instances_root.iterdir() if p.is_dir())
    if args.max_trajectories > 0:
        instances = instances[: args.max_trajectories]

    rows: list[dict[str, Any]] = []
    used = 0
    for instance in instances:
        batch = build_instance(instance)
        if batch:
            rows.extend(batch)
            used += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"[webarena_offline] trajectories={used}/{len(instances)} "
        f"steps={len(rows)} output={args.output}"
    )


if __name__ == "__main__":
    main()
