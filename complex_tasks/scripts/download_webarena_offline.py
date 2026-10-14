#!/usr/bin/env python3
"""Download text-only WebArena interaction traces from an HF mirror."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests


DEFAULT_REPO = "open-social-world/autolibra"
DEFAULT_ENDPOINT = "https://hf-mirror.com"


def get_json(session: requests.Session, url: str, timeout: int) -> Any:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def tree_url(endpoint: str, repo: str, path: str) -> str:
    encoded = "/".join(quote(part, safe="") for part in path.split("/"))
    return f"{endpoint}/api/datasets/{repo}/tree/main/{encoded}?recursive=true&expand=false"


def resolve_url(endpoint: str, repo: str, path: str) -> str:
    encoded = "/".join(quote(part, safe="") for part in path.split("/"))
    return f"{endpoint}/datasets/{repo}/resolve/main/{encoded}"


def download_one(
    endpoint: str,
    repo: str,
    remote_path: str,
    output: Path,
    timeout: int,
) -> tuple[str, bool]:
    relative = Path(remote_path).relative_to("webarena")
    destination = output / "webarena" / relative
    if destination.is_file() and destination.stat().st_size > 0:
        return remote_path, False
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(resolve_url(endpoint, repo, remote_path), timeout=timeout)
    response.raise_for_status()
    # Validate before committing the file so interrupted runs remain resumable.
    json.loads(response.text)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_text(response.text, encoding="utf-8")
    temporary.replace(destination)
    return remote_path, True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--max-trajectories", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    endpoint = args.endpoint.rstrip("/")
    session = requests.Session()
    root = get_json(session, tree_url(endpoint, args.repo, "webarena/instances"), args.timeout)
    instances = sorted(
        item["path"] for item in root
        if item.get("type") == "directory"
        and item.get("path", "").count("/") == 2
    )
    if args.max_trajectories > 0:
        instances = instances[: args.max_trajectories]
    print(f"[download_webarena] listing {len(instances)} trajectories", flush=True)

    remote_files: list[str] = []
    for index, instance in enumerate(instances, 1):
        for role in ("agent", "user"):
            path = f"{instance}/{role}/json_data"
            listing = get_json(session, tree_url(endpoint, args.repo, path), args.timeout)
            remote_files.extend(
                item["path"] for item in listing
                if item.get("type") == "file" and item.get("path", "").endswith(".json")
            )
        if index % 50 == 0 or index == len(instances):
            print(f"[download_webarena] listed {index}/{len(instances)}", flush=True)

    downloaded = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                download_one, endpoint, args.repo, path, args.output, args.timeout
            ): path
            for path in remote_files
        }
        for index, future in enumerate(as_completed(futures), 1):
            path = futures[future]
            try:
                _path, changed = future.result()
                downloaded += int(changed)
            except Exception as exc:
                raise RuntimeError(f"Failed to download {path}: {exc}") from exc
            if index % 250 == 0 or index == len(futures):
                print(
                    f"[download_webarena] files {index}/{len(futures)} new={downloaded}",
                    flush=True,
                )
    print(f"[download_webarena] complete output={args.output}", flush=True)


if __name__ == "__main__":
    main()
