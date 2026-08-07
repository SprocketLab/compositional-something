#!/usr/bin/env python3
"""Download and pin Spider 1.0, SParC, and the test-suite databases.

SParC plan §7.1: pin versions and hashes.  All three artifacts are distributed
as Google Drive zips (Yale LILY pages / taoyds/test-suite-sql-eval README);
the Drive file ids live in data/sparc_pins.json.  The first successful
download freezes sha256 + content counts into data/sparc_manifest.json
(tracked); every later run asserts against the manifest, so a silent
re-upload upstream becomes a hard failure here.

Fallback when gdown hits a Drive quota/cookie wall: download the zips in a
browser, scp them anywhere, and pass --local-zip spider=/path/to/spider.zip
(repeatable).  Verification is identical for both routes.

Run on the login node (network + CPU only):
  PYTHONPATH=. $PY reports/composition_screen/sparc_fetch.py \
      --pins reports/composition_screen/data/sparc_pins.json \
      --out-root reports/composition_screen/data/sparc_raw
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ARTIFACTS = ("spider", "sparc", "testsuite")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_zip(name: str, pin: dict, out_root: Path, local: dict[str, Path]) -> Path:
    zip_path = out_root / f"{name}.zip"
    if zip_path.exists():
        print(f"{name}: zip present, skipping download", flush=True)
        return zip_path
    if name in local:
        print(f"{name}: copying local zip {local[name]}", flush=True)
        shutil.copyfile(local[name], zip_path)
        return zip_path
    import gdown
    print(f"{name}: gdown id={pin['gdrive_id']}", flush=True)
    got = gdown.download(id=pin["gdrive_id"], output=str(zip_path), quiet=False)
    if got is None or not zip_path.exists():
        raise SystemExit(
            f"{name}: gdown failed -- download in a browser from "
            f"{pin['source']} and rerun with --local-zip {name}=/path/to.zip")
    return zip_path


def unpack(name: str, zip_path: Path, out_root: Path) -> Path:
    dest = out_root / name
    if dest.exists():
        print(f"{name}: unpacked dir present, skipping extract", flush=True)
        return dest
    tmp = out_root / f"_{name}_extract"
    if tmp.exists():
        shutil.rmtree(tmp)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)
    # Zips may wrap everything in one top-level directory; normalize that away.
    entries = [p for p in tmp.iterdir() if not p.name.startswith("__MACOSX")]
    if len(entries) == 1 and entries[0].is_dir():
        entries[0].rename(dest)
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        tmp.rename(dest)
    return dest


def count_json(path: Path) -> int:
    return len(json.loads(path.read_text()))


def collect_counts(name: str, root: Path) -> dict:
    if name == "spider":
        counts = {f: count_json(root / f"{f}.json")
                  for f in ("train_spider", "train_others", "dev")
                  if (root / f"{f}.json").exists()}
        counts["database_dirs"] = sum(
            1 for p in (root / "database").iterdir() if p.is_dir())
        counts["tables"] = count_json(root / "tables.json")
        return counts
    if name == "sparc":
        counts = {f: count_json(root / f"{f}.json")
                  for f in ("train", "dev") if (root / f"{f}.json").exists()}
        counts["tables"] = count_json(root / "tables.json")
        db = root / "database"
        counts["database_dirs"] = (
            sum(1 for p in db.iterdir() if p.is_dir()) if db.exists() else 0)
        return counts
    # testsuite: a tree of sqlite files, layout differs per release
    sqlites = list(root.rglob("*.sqlite"))
    return {"sqlite_files": len(sqlites),
            "db_dirs": len({p.parent.name for p in sqlites})}


def assert_counts(name: str, counts: dict) -> None:
    published = {
        "spider": {"train_spider": 7000, "train_others": 1659, "dev": 1034},
        "sparc": {"train": 3034, "dev": 422},
    }.get(name, {})
    for key, want in published.items():
        got = counts.get(key)
        if got != want:
            raise SystemExit(
                f"{name}: {key} count {got} != published {want} -- "
                "release changed upstream; re-pin deliberately or fix the download")
    if name == "spider" and counts["database_dirs"] < 160:
        raise SystemExit(f"spider: only {counts['database_dirs']} database dirs")
    if name == "testsuite" and counts["sqlite_files"] == 0:
        raise SystemExit("testsuite: no sqlite files found after extraction")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pins", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--local-zip", action="append", default=[],
                    metavar="NAME=PATH")
    args = ap.parse_args()

    pins = json.loads(args.pins.read_text())
    local = {}
    for spec in args.local_zip:
        name, _, path = spec.partition("=")
        local[name] = Path(path)
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.pins.parent / "sparc_manifest.json"
    manifest = (json.loads(manifest_path.read_text())
                if manifest_path.exists() else {})

    pinned_now = False
    for name in ARTIFACTS:
        pin = pins[name]
        zip_path = fetch_zip(name, pin, args.out_root, local)
        digest = sha256_file(zip_path)
        if pin["sha256"] is None:
            pin["sha256"] = digest
            pinned_now = True
            print(f"{name}: PINNED sha256={digest}", flush=True)
        elif pin["sha256"] != digest:
            raise SystemExit(
                f"{name}: sha256 mismatch\n  pinned {pin['sha256']}\n  got    {digest}")
        else:
            print(f"{name}: sha256 verified", flush=True)
        root = unpack(name, zip_path, args.out_root)
        counts = collect_counts(name, root)
        assert_counts(name, counts)
        manifest[name] = {"sha256": digest, "bytes": zip_path.stat().st_size,
                          "counts": counts, "gdrive_id": pin["gdrive_id"]}
        print(f"{name}: counts {counts}", flush=True)

    if pinned_now:
        args.pins.write_text(json.dumps(pins, indent=2) + "\n")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest written to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
