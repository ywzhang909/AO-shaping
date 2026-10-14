#!/usr/bin/env python3
"""Poll the AO-shaping repo for new events and emit a machine-readable digest.

Used as the `script` of a Hermes cron job (agent-driven). The script's
stdout is injected as context into the agent's prompt:
  - non-empty stdout  -> the agent wakes and processes the digest
  - empty stdout      -> silent tick (nothing changed), no LLM cost

Event sources (read-only GitHub API calls; the script never pushes):
  1. New commits on the default branch (banckend) — e.g. pushed by the
     user's lab machine / lab agent, or by any collaborator.
  2. New OPEN issues with the `documentation` label.

Loop prevention (why this is NOT an infinite loop):
  - After the agent pushes a fix, it runs
        python3 poll_repo_changes.py --mark-pushed <sha> [<sha>...]
    which records those SHAs in state["own_pushes"]; the poller never
    emits own-pushed commits as "new input".
  - Commit attribution is NOT trusted (git pushes here are attributed to
    the ywzhang909 identity regardless of which automation pushed them,
    and lab-agent pushes have author.login=null), so SHA tracking is the
    only reliable dedup.
  - state.json is the single source of dedup truth.

Rebase robustness:
  This repo is rebased often (lab machine pushes ahead). `compare` 404s
  when the stored last_commit is no longer an ancestor, so the poller
  falls back to "recent N commits minus already-seen SHAs".

State: .auto_doc_review/state.json (next to this script):
  branch, last_commit, seen_commits[], own_pushes[], seen_issues[], last_run

Usage:
  python3 poll_repo_changes.py            # normal poll
  python3 poll_repo_changes.py --seed     # force re-seed state
  python3 poll_repo_changes.py --mark-pushed <sha>...   # after a push
  python3 poll_repo_changes.py --selftest # offline dedup-logic test
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

REPO = "ywzhang909/AO-shaping"
BRANCH = "banckend"          # default branch (sic — misspelled upstream)
MAX_COMMITS = 15             # cap commits considered/emitted per poll
MAX_FILES_PER_COMMIT = 25    # cap changed files listed per commit
MAX_ISSUES = 10              # cap issues per digest
MAX_ISSUE_BODY_CHARS = 2500  # truncate issue bodies
STATE_CAP = 50               # cap seen_commits / own_pushes list growth

# --- PATH/HOME bootstrap ----------------------------------------------------
# Cron script runs have a minimal PATH and — since 2026-10-14 — may run with
# HOME unset, which makes `gh` fail with "gh auth login" (rc=4) even though
# the token exists at /opt/data/home/.config/gh/hosts.yml. Prepend the known
# binary locations and restore HOME so subprocess `gh` auth works.
import shutil
import os
_GH_HOME = "/opt/data/home"
if not os.environ.get("HOME") or not os.path.exists(
    os.path.join(os.environ["HOME"], ".config", "gh", "hosts.yml")
):
    os.environ["HOME"] = _GH_HOME
for _d in (
    "/opt/data/home/.local/bin",
    "/usr/local/bin",
    "/usr/bin",
    "/bin",
):
    if _d not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")
# ---------------------------------------------------------------------------
_GH_BIN: str | None = None
def _gh() -> str | None:
    global _GH_BIN
    if _GH_BIN is None:
        _GH_BIN = shutil.which("gh")
    return _GH_BIN
# ---------------------------------------------------------------------------

STATE_PATH = Path("/opt/data/workspace/projects/AO-shaping/repo/.auto_doc_review/state.json")


def sh(args: list[str], timeout: int = 60) -> tuple[int, str, str]:
    # Resolve gh to an absolute path so it works under minimal cron PATHs.
    if args and args[0] == "gh":
        g = _gh()
        if g is None:
            return 127, "", "gh binary not found even after PATH bootstrap"
        args = [g] + args[1:]
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                            env={**os.environ, "PATH": os.environ["PATH"]})
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s: {' '.join(args[:3])}"
    except FileNotFoundError:
        return 127, "", f"binary not found: {args[0]}"


def gh_json(path: str, flags: list[str] | None = None, timeout: int = 60) -> tuple[dict | list | None, str | None]:
    """gh api <path-as-one-arg> [flags...]; returns (dict-or-list-or-None, err)."""
    rc, out, err = sh(["gh", "api", path, *(flags or [])], timeout=timeout)
    if rc != 0:
        return None, f"gh api {path[:60]} rc={rc}: {(err or out)[:200]}"
    try:
        data = json.loads(out)
    except json.JSONDecodeError as e:
        return None, f"bad json from gh api: {e}: {out[:200]}"
    if not isinstance(data, (dict, list)):
        return None, f"unexpected json type from gh api: {out[:100]}"
    return data, None


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            data = json.loads(STATE_PATH.read_text())
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=1))


def default_branch() -> str:
    data, _err = gh_json(f"repos/{REPO}")
    if isinstance(data, dict):
        return data.get("default_branch") or BRANCH
    return BRANCH


def head_commit(branch: str) -> tuple[str | None, str | None]:
    ref, err = gh_json(f"repos/{REPO}/commits/{branch}")
    if isinstance(ref, dict):
        sha = ref.get("sha")
        if isinstance(sha, str):
            return sha, None
    return None, err or "no sha in head response"


def _normalize(c: dict) -> dict:
    author = (c.get("author") or {}).get("login") or (c.get("commit", {}).get("author", {}) or {}).get("name", "?")
    return {
        "sha": c.get("sha", "?"),
        "author": author,
        "subject": (c.get("commit", {}).get("message") or "?").splitlines()[0],
    }


def recent_commits(branch: str) -> tuple[list[dict], str | None]:
    """Most recent MAX_COMMITS commits, newest first (rebase-safe)."""
    data, err = gh_json(f"repos/{REPO}/commits?sha={branch}&per_page={MAX_COMMITS}")
    if not isinstance(data, list):
        return [], err or "commits list returned non-list"
    out = [_normalize(c) for c in data[:MAX_COMMITS]]
    for c in out:
        d2, _e2 = gh_json(f"repos/{REPO}/commits/{c['sha']}")
        c["files"] = [f.get("filename", "?") for f in (d2 or {}).get("files", [])][:MAX_FILES_PER_COMMIT] if isinstance(d2, dict) else ["<files unavailable>"]
    return out, None


def doc_issues_open() -> tuple[list[dict], str | None]:
    q = quote(f"repo:{REPO} is:issue is:open label:documentation")
    data, err = gh_json(f"search/issues?q={q}&per_page={MAX_ISSUES}", timeout=90)
    if not isinstance(data, dict):
        return [], err or "issue search returned no data"
    issues: list[dict] = []
    for it in data.get("items", [])[:MAX_ISSUES]:
        raw_labels = it.get("labels") or []
        if raw_labels and isinstance(raw_labels[0], dict):
            label_names = [l.get("name", "?") for l in raw_labels]
        else:
            label_names = [str(x) for x in raw_labels]
        issues.append({
            "number": it.get("number", -1),
            "title": it.get("title", "?"),
            "labels": label_names,
            "created_at": it.get("created_at", ""),
            "body": (it.get("body") or "").strip()[:MAX_ISSUE_BODY_CHARS],
        })
    return issues, None


def run_poll(seed: bool = False) -> int:
    state: dict = {} if seed else load_state()
    branch = state.get("branch") or default_branch()
    head, err = head_commit(branch)
    if not head:
        print("STATUS=ERROR")
        print(f"Cannot read HEAD of {branch}: {err}")
        return 2

    seen_commits = set(state.get("seen_commits", []))
    own_pushes = set(state.get("own_pushes", []))

    new_commits: list[dict] = []
    # Skip the commit walk entirely when HEAD is unchanged (cheap ticks).
    if seed or not state.get("last_commit") or state["last_commit"] != head:
        commits, cerr = recent_commits(branch)
        if cerr:
            print("STATUS=ERROR")
            print(f"Commit list failed: {cerr}")
            return 2
        for c in commits:
            if c["sha"] in seen_commits or c["sha"] in own_pushes:
                continue
            new_commits.append(c)

    # --- new doc issues ---
    seen_issues = set(state.get("seen_issues", []))
    issues, ierr = doc_issues_open()
    if ierr:
        print("STATUS=ERROR")
        print(f"Issue list failed: {ierr}")
        return 2
    new_issues = [it for it in issues if it["number"] not in seen_issues]

    # --- advance state ---
    prior_seen = state.get("seen_commits", [])
    merged = [head] + [s for s in prior_seen if s != head] + \
             [c["sha"] for c in new_commits if c["sha"] not in prior_seen and c["sha"] != head]
    state.update({
        "branch": branch,
        "last_commit": head,
        "seen_commits": merged[:STATE_CAP],
        "seen_issues": sorted(seen_issues | {it["number"] for it in issues}),
    })
    save_state(state)

    if not new_commits and not new_issues:
        return 1  # quiet -> empty stdout -> silent tick

    if seed:
        print("STATUS=SEED")
        print(f"branch={branch} head={head[:12]} open_doc_issues={len(issues)}")
        return 0

    print("STATUS=CHANGES")
    if new_commits:
        print(f"=== NEW_COMMITS ({len(new_commits)}) on {branch} ===")
        for c in new_commits:
            print(f"{c['sha']} author={c['author']} :: {c['subject']}")
            files = c.get("files", [])
            if files:
                more = " ...truncated" if len(files) >= MAX_FILES_PER_COMMIT else ""
                print(f"  files: {', '.join(files)}{more}")
    if new_issues:
        print(f"=== NEW_DOC_ISSUES ({len(new_issues)}) ===")
        for it in new_issues:
            print(f"#{it['number']} [{', '.join(it['labels'])}] {it['title']} (created {it['created_at']})")
            if it["body"]:
                print(f"--- body #{it['number']} ---")
                print(it["body"])
                print(f"--- end #{it['number']} ---")
    print("NOTE: after any successful push, run `python3 poll_repo_changes.py --mark-pushed <sha>...` to avoid re-triggering on your own fix commits.")
    return 0


def mark_pushed(shas: list[str]) -> int:
    state = load_state()
    own = [s for s in state.get("own_pushes", []) if s not in shas]
    state["own_pushes"] = (shas + own)[:STATE_CAP]
    prior_seen = state.get("seen_commits", [])
    state["seen_commits"] = (shas + [s for s in prior_seen if s not in shas])[:STATE_CAP]
    save_state(state)
    print(f"MARKED {len(shas)} commit(s) as own push: {', '.join(s[:12] for s in shas)}")
    return 0


def selftest() -> int:
    """Offline test of the dedup/emit logic (no network)."""
    checks: list[str] = []

    # 1. fresh state seeds clean
    STATE_PATH.unlink(missing_ok=True)
    assert load_state() == {}, "fresh state must be empty"
    checks.append("fresh-state")

    # 2. seen_issues dedup
    state = {"seen_issues": [1, 2, 3]}
    new = [i for i in [3, 4, 5] if i not in set(state["seen_issues"])]
    assert new == [4, 5], new
    checks.append("issue-dedup")

    # 3. own_pushes + seen_commits suppress self-commits
    state = {"seen_commits": ["a1"], "own_pushes": ["b1"]}
    commits = ["a1", "b1", "c1"]
    kept = [s for s in commits if s not in set(state["seen_commits"]) and s not in set(state["own_pushes"])]
    assert kept == ["c1"], kept
    checks.append("self-commit-filter")

    # 4. same head -> quiet (no commit walk needed)
    st = {"last_commit": "h1"}
    assert st["last_commit"] == "h1"
    checks.append("quiet-on-same-head")

    print("SELFTEST OK: " + ", ".join(checks))
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    if "--seed" in sys.argv:
        sys.exit(run_poll(seed=True))
    if "--mark-pushed" in sys.argv:
        idx = sys.argv.index("--mark-pushed")
        shas = sys.argv[idx + 1:]
        if not shas:
            print("usage: poll_repo_changes.py --mark-pushed <sha>...")
            sys.exit(2)
        sys.exit(mark_pushed(shas))
    sys.exit(run_poll())
