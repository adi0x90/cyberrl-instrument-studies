#!/usr/bin/env python3
"""Offline dry-run tests for the three runners, the report and the statistics.

What is asserted here:
  1. the pure statistics helpers (Wilson, McNemar, Fisher, kappa) against values that
     were verified against scipy at build time -- these tests need nothing but Python;
  2. each runner's `--mock --limit 2` contract: JSONL schema, manifest fields,
     resumability with no duplicate unit keys, and no API key in the environment;
  3. analysis/summarize.py turns the mock artifacts into a REPORT.md carrying the
     three endpoint tables plus the limits and provenance sections.

If `harness/` or `tasks/` are not importable, the runner/report tests skip with
the exact missing module while the pure statistics tests still run.

Run with:  python -m pytest -q tests
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUMMARIZE = ROOT / "analysis" / "summarize.py"

RUNNERS = {"E1": "run_e1.py", "E2": "run_e2.py", "E3": "run_e3.py"}

DEFAULT_MODELS = (
    "tencent/hy4-flash",
    "z-ai/glm-5.3-flash",
    "deepseek/deepseek-v4.1-flash-1017",
)

REQUIRED_COMMON = (
    "unit",
    "experiment",
    "task_id",
    "model",
    "decoding",
    "admitted",
    "admission_reason",
    "raw_expression",
    "verdict",
    "reasons",
    "prompt_tokens",
    "completion_tokens",
    "http_status",
    "ts_utc",
    "seed",
)

CLI_FLAGS = ("--mock", "--dry-run", "--limit", "--models", "--out", "--resume")


# --- load the analysis module without requiring a package -------------------


def load_summarize():
    spec = importlib.util.spec_from_file_location("summarize_under_test", SUMMARIZE)
    assert spec is not None and spec.loader is not None, f"cannot load {SUMMARIZE}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["summarize_under_test"] = module
    spec.loader.exec_module(module)
    return module


SUMM = load_summarize()


# --- is the harness ready? -------------------------------------------------


def _harness_gap() -> str | None:
    """Return a human-readable reason if harness/ or tasks/ are missing."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    modules = (
        "harness.models",
        "harness.envelopes",
        "harness.tasks_io",
        "harness.graders",
        "harness.polarity",
    )
    for name in modules:
        try:
            __import__(name)
        except Exception as exc:  # noqa: BLE001 - report whatever is missing
            return f"{name} not importable ({type(exc).__name__}: {exc})"
    task_files = sorted((ROOT / "tasks").glob("*.json"))
    if not task_files:
        return f"no task files under {ROOT / 'tasks'}"
    return None


HARNESS_GAP = _harness_gap()
needs_harness = pytest.mark.skipif(
    HARNESS_GAP is not None,
    reason=(
        "harness/ or tasks/ not importable "
        f"({HARNESS_GAP}); this test requires real harness behaviour, no stub is used"
    ),
)


# --- test environment ------------------------------------------------------


def clean_env() -> dict[str, str]:
    """Environment for the child processes.

    OPENROUTER_API_KEY is dropped by *name* so mock mode is proven to need no key;
    its value is never read, logged, or written.
    """
    env = {key: value for key, value in os.environ.items() if key != "OPENROUTER_API_KEY"}
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.pop("PYTHONWARNINGS", None)
    return env


def run_python(script: str, args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    command = [sys.executable, str(ROOT / script)] + args
    return subprocess.run(
        command,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def manifest_for(path: Path) -> Path:
    return path.with_name(path.stem + ".manifest.json")


def read_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


# --- 1. statistics (no harness needed) -------------------------------------


@pytest.mark.parametrize(
    "k,n,low,high",
    [
        (5, 20, 0.11186, 0.468705),
        (0, 10, 0.0, 0.27754),
        (10, 10, 0.72246, 1.0),
    ],
)
def test_wilson_ci_matches_reference(k, n, low, high):
    got_low, got_high = SUMM.wilson_ci(k, n)
    assert got_low == pytest.approx(low, abs=5e-5)
    assert got_high == pytest.approx(high, abs=5e-5)
    assert 0.0 <= got_low <= got_high <= 1.0


def test_wilson_ci_handles_empty_sample():
    assert SUMM.wilson_ci(0, 0) == (0.0, 0.0)
    assert SUMM.wilson_ci(3, 0) == (0.0, 0.0)


@pytest.mark.parametrize(
    "b,c,expected",
    [
        (0, 0, 1.0),
        (10, 0, 0.001953125),
        (1, 4, 0.375),
        (7, 1, 0.0703125),
        (3, 3, 1.0),
    ],
)
def test_mcnemar_exact_matches_reference(b, c, expected):
    assert SUMM.mcnemar_exact(b, c) == pytest.approx(expected, abs=1e-12)
    assert SUMM.mcnemar_exact(b, c) == SUMM.mcnemar_exact(c, b)


@pytest.mark.parametrize(
    "table,expected",
    [
        ((3, 1, 1, 3), 0.4857142857142857),
        ((8, 2, 1, 5), 0.03496503496503496),
        ((12, 0, 2, 10), 6.730380939561179e-05),
        ((5, 5, 5, 5), 1.0),
        ((1, 9, 11, 3), 0.002759456185220083),
    ],
)
def test_fisher_exact_matches_reference(table, expected):
    got = SUMM.fisher_exact(*table)
    assert got == pytest.approx(expected, rel=1e-9)
    assert SUMM.fisher_exact(table[1], table[0], table[3], table[2]) == pytest.approx(
        expected, rel=1e-9
    )


def test_fisher_exact_degenerate_tables():
    assert SUMM.fisher_exact(0, 0, 0, 0) == 1.0
    assert 0.0 <= SUMM.fisher_exact(1, 0, 0, 1) <= 1.0


def test_kappa_perfect_agreement():
    assert SUMM.kappa([(True, True), (True, True), (False, False)]) == pytest.approx(1.0)


def test_kappa_known_value():
    # observed agreement 0.75, expected 0.5 -> (0.75 - 0.5) / (1 - 0.5) = 0.5
    pairs = [(True, True), (True, True), (True, False), (False, False)]
    assert SUMM.kappa(pairs) == pytest.approx(0.5)


def test_kappa_chance_level_and_degenerate_inputs():
    assert SUMM.kappa([(True, True), (True, False), (False, True), (False, False)]) == pytest.approx(0.0)
    assert SUMM.kappa([]) != SUMM.kappa([])  # nan
    assert SUMM.kappa([(True, True), (True, True)]) == pytest.approx(1.0)


def load_runner(filename: str):
    """Import a runner module by path (the runners are scripts, not a package)."""
    name = filename.replace(".py", "")
    spec = importlib.util.spec_from_file_location(f"runner_{name}", ROOT / filename)
    assert spec is not None and spec.loader is not None, f"cannot load {filename}"
    module = importlib.util.module_from_spec(spec)
    # Register before exec: dataclasses resolve their module through sys.modules.
    sys.modules[f"runner_{name}"] = module
    spec.loader.exec_module(module)
    return module


# --- 2. E2 candidate generation (exact rules from docs/DESIGN.md §4) ------------


@needs_harness
def test_e2_mutations_are_exact_and_parse_checked():
    e2 = load_runner("run_e2.py")
    # swap the first comparison operator, in place
    assert e2.swap_comparison("a == b") == "a != b"
    assert e2.swap_comparison("not (a < b)") == "not (a >= b)"
    assert e2.swap_comparison("a >= b") == "a < b"
    assert e2.swap_comparison("False") is None          # nothing to swap
    # invert the first membership / identity test
    assert e2.invert_membership("x in y") == "x not in y"
    assert e2.invert_membership("x not in y") == "x in y"
    assert e2.invert_membership("x is not None") == "x is None"
    assert e2.invert_membership("x is None") == "x is not None"
    assert e2.invert_membership("a == b") is None
    # weaken with a tautology
    assert e2.weaken_guard("a == b") == "(a == b) or True"
    # the parse filter is what keeps only well-formed mutations
    assert e2.parses_as_expression("a and b")
    assert not e2.parses_as_expression("a and")
    assert not e2.parses_as_expression("")
    assert not e2.parses_as_expression("x = 1")


@needs_harness
def test_e2_seeded_candidate_families():
    e2 = load_runner("run_e2.py")
    task = {"seeded_bad": [{"family": "symptom_suppression", "expression": "a == b", "why": "blocks the PoC only"}]}
    kinds = [candidate.kind for candidate in e2.seeded_candidates(task)]
    assert kinds == ["seeded", "mutation:swap_op", "mutation:weaken_guard"]
    assert all(candidate.seeded_family == "symptom_suppression" for candidate in e2.seeded_candidates(task))

    in_task = {"seeded_bad": [{"family": "guard_too_weak", "expression": "x in y"}]}
    kinds = [candidate.kind for candidate in e2.seeded_candidates(in_task)]
    assert kinds == ["seeded", "mutation:invert_membership", "mutation:weaken_guard"]

    # duplicates are collapsed, so no unit key can collide
    dup_task = {"seeded_bad": [
        {"family": "a", "expression": "x == y"},
        {"family": "a", "expression": "x == y"},
    ]}
    assert len(e2.seeded_candidates(dup_task)) == 3


@needs_harness
def test_e2_adversarial_shapes():
    e2 = load_runner("run_e2.py")
    assert [text for text, _ in e2._adversarial_expressions("a == b")] == ["a == b"]
    assert [text for text, _ in e2._adversarial_expressions({"expression": "x", "why": "keeps root cause"})] == ["x"]
    assert [text for text, _ in e2._adversarial_expressions(["x", {"expression": "y"}, 42, {}])] == ["x", "y"]
    assert e2._adversarial_expressions(None) == []
    assert e2._adversarial_expressions({"no_expression": True}) == []

    task = {"seeded_bad": [], "adversarial": [{"expression": "True", "why": "satisfies obligations"}]}
    candidates = e2.adversarial_candidates(task)
    assert len(candidates) == 1
    assert candidates[0].kind == "adversarial"
    assert candidates[0].expression == "True"
    assert "satisfies obligations" in candidates[0].detail


# --- 3. runner contracts, mock mode ---------------------------------------


@pytest.fixture(scope="session")
def dryrun(tmp_path_factory):
    """Run each runner twice (fresh, then --resume) and build the mock report."""
    if HARNESS_GAP is not None:
        pytest.skip(f"harness/ or tasks/ not importable: {HARNESS_GAP}")

    work = tmp_path_factory.mktemp("dryrun")
    env = clean_env()
    artifacts: dict[str, dict] = {}

    for experiment, script in RUNNERS.items():
        jsonl = work / f"{experiment.lower()}.jsonl"
        args = ["--mock", "--limit", "2", "--out", str(jsonl)]
        if experiment == "E2":
            # Hermetic: import patches from this temp E1 run, not from whatever
            # happens to be in results/ on the developer's machine.
            args.extend(["--e1", str(work / "e1.jsonl")])
        first = run_python(script, args, env)
        assert first.returncode == 0, (
            f"{script} --mock --limit 2 failed (rc={first.returncode})\n"
            f"stdout:\n{first.stdout}\nstderr:\n{first.stderr}"
        )
        first_text = jsonl.read_text(encoding="utf-8")
        # The manifest is rewritten by the resume run, so snapshot the fresh-run
        # manifest before the second run clobbers it.
        first_manifest = json.loads(manifest_for(jsonl).read_text(encoding="utf-8"))

        second = run_python(script, args + ["--resume"], env)
        assert second.returncode == 0, (
            f"{script} --mock --limit 2 --resume failed (rc={second.returncode})\n"
            f"stdout:\n{second.stdout}\nstderr:\n{second.stderr}"
        )
        artifacts[experiment] = {
            "jsonl": jsonl,
            "manifest": manifest_for(jsonl),
            "first": first_text,
            "second": jsonl.read_text(encoding="utf-8"),
            "manifest_first": first_manifest,
            "env": env,
        }

    report = work / "REPORT.md"
    summarize = run_python(
        "analysis/summarize.py",
        [
            "--mock",
            "--e1",
            str(work / "e1.jsonl"),
            "--e2",
            str(work / "e2.jsonl"),
            "--e3",
            str(work / "e3.jsonl"),
            "--out",
            str(report),
        ],
        env,
    )
    assert summarize.returncode == 0, (
        f"summarize failed (rc={summarize.returncode})\n"
        f"stdout:\n{summarize.stdout}\nstderr:\n{summarize.stderr}"
    )
    artifacts["report"] = {"path": report, "env": env}
    return artifacts


@needs_harness
@pytest.mark.parametrize("experiment", sorted(RUNNERS))
def test_jsonl_schema(dryrun, experiment):
    path = dryrun[experiment]["jsonl"]
    assert path.exists(), f"{path} was not written"
    records = read_records(path)
    assert records, f"{path} has no records"

    for record in records:
        missing = [key for key in REQUIRED_COMMON if key not in record]
        assert not missing, f"record {record.get('unit')!r} missing keys {missing}"
        assert record["experiment"] == experiment
        assert record["model"] in DEFAULT_MODELS or record["model"] == "none"
        assert isinstance(record["reasons"], list)
        assert isinstance(record["admitted"], bool)
        assert isinstance(record["ts_utc"], str) and record["ts_utc"].endswith("Z")
        assert isinstance(record["prompt_tokens"], int)
        assert isinstance(record["completion_tokens"], int)
        assert isinstance(record["http_status"], int)
        if experiment == "E1":
            assert "envelope" in record and record["envelope"] in ("strict", "fence", "toolcall")
            assert record["decoding"] in (0, 1)
        if experiment == "E3":
            assert "variant" in record and record["variant"] in ("explicit", "neutral")
            assert record["decoding"] in (0, 1, 2)
        if experiment == "E2":
            assert "candidate_kind" in record
            assert "weak_verdict" in record


@needs_harness
@pytest.mark.parametrize("experiment", sorted(RUNNERS))
def test_unit_keys_unique(dryrun, experiment):
    units = [record["unit"] for record in read_records(dryrun[experiment]["jsonl"])]
    assert len(units) == len(set(units)), "duplicate unit keys inside one run"


@needs_harness
@pytest.mark.parametrize("experiment", sorted(RUNNERS))
def test_manifest_fields(dryrun, experiment):
    path = dryrun[experiment]["manifest"]
    assert path.exists(), f"{path} was not written"
    # Fresh-run snapshot: the on-disk manifest now describes the resume run, where
    # records_written is legitimately 0.
    manifest = dryrun[experiment]["manifest_first"]

    for key in (
        "experiment",
        "git_sha",
        "utc",
        "models",
        "task_sha256",
        "argv",
        "seed",
        "mock",
        "status",
        "records_written",
        "records_skipped",
    ):
        assert key in manifest, f"manifest missing {key!r}"

    assert manifest["experiment"] == experiment
    assert isinstance(manifest["git_sha"], str) and manifest["git_sha"]
    assert isinstance(manifest["utc"], str) and manifest["utc"].endswith("Z")
    assert manifest["models"] == list(DEFAULT_MODELS)
    assert isinstance(manifest["task_sha256"], dict)
    assert all(len(digest) == 64 for digest in manifest["task_sha256"].values())
    assert isinstance(manifest["argv"], list) and "--mock" in manifest["argv"]
    assert manifest["seed"] == 20261120
    assert manifest["mock"] is True
    assert manifest["status"] == "complete"
    assert manifest["records_written"] > 0


@needs_harness
@pytest.mark.parametrize("experiment", sorted(RUNNERS))
def test_resume_is_idempotent(dryrun, experiment):
    """A second run with --resume must add no records and duplicate no key."""
    artifact = dryrun[experiment]
    assert artifact["second"] == artifact["first"], "resume rewrote the JSONL"
    records = read_records(artifact["jsonl"])
    units = [record["unit"] for record in records]
    assert len(units) == len(set(units))

    manifest = json.loads(artifact["manifest"].read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["records_written"] == 0, "resume re-wrote records it should have skipped"
    assert manifest["records_skipped"] == len(records)


@needs_harness
@pytest.mark.parametrize("experiment", ["E1", "E2", "E3"])
def test_cli_flags_and_no_key_needed(dryrun, experiment):
    script = RUNNERS[experiment]
    env = dryrun[experiment]["env"]
    assert "OPENROUTER_API_KEY" not in env

    help_run = run_python(script, ["--help"], env)
    assert help_run.returncode == 0, help_run.stderr
    for flag in CLI_FLAGS:
        assert flag in help_run.stdout, f"{script} --help does not document {flag}"


@needs_harness
def test_resume_detects_config_drift(tmp_path, dryrun):
    """Resuming under a different seed must warn and record the drift, not silently pool."""
    env = dryrun["E1"]["env"]
    jsonl = tmp_path / "seeded_e1.jsonl"
    first = run_python(
        "run_e1.py", ["--mock", "--limit", "1", "--seed", "7", "--out", str(jsonl)], env
    )
    assert first.returncode == 0, first.stderr
    second = run_python(
        "run_e1.py",
        ["--mock", "--limit", "1", "--seed", "99", "--out", str(jsonl), "--resume"],
        env,
    )
    assert second.returncode == 0, second.stderr
    assert "configuration changed" in second.stderr
    manifest = json.loads(manifest_for(jsonl).read_text(encoding="utf-8"))
    assert manifest["config_drift"], "config drift was not recorded in the manifest"
    assert "seed" in manifest["config_drift"][0]
    assert manifest["records_written"] == 0  # nothing re-attempted under the new seed


@needs_harness
def test_dry_run_alias_and_limit_validation(dryrun, tmp_path):
    """--dry-run is an alias for --mock, and --limit 0 is rejected before any work."""
    env = dryrun["E1"]["env"]
    jsonl = tmp_path / "alias_e1.jsonl"
    result = run_python("run_e1.py", ["--dry-run", "--limit", "1", "--out", str(jsonl)], env)
    assert result.returncode == 0, result.stderr
    manifest = json.loads(manifest_for(jsonl).read_text(encoding="utf-8"))
    assert manifest["mock"] is True
    assert manifest["limit"] == 1
    assert len(read_records(jsonl)) == 1 * 3 * 3 * 2  # task x envelope x model x decoding

    rejected = run_python("run_e1.py", ["--mock", "--limit", "0", "--out", str(tmp_path / "z.jsonl")], env)
    assert rejected.returncode != 0
    assert "--limit" in (rejected.stderr + rejected.stdout)


@needs_harness
def test_real_mode_without_key_fails_fast(dryrun, tmp_path):
    """Without --mock and without the env var, the runner must stop before any call."""
    target = tmp_path / "real_e1.jsonl"
    result = run_python("run_e1.py", ["--limit", "1", "--out", str(target)], dryrun["E1"]["env"])
    assert result.returncode != 0
    assert "OPENROUTER_API_KEY" in (result.stderr + result.stdout)
    assert not target.exists(), "the runner created an output file before failing fast"


# --- 4. report -------------------------------------------------------------


def _section_body(text: str, heading: str) -> str:
    assert heading in text, f"report is missing heading {heading!r}"
    tail = text.split(heading, 1)[1]
    for stop in ("\n### ", "\n## "):
        if stop in tail:
            tail = tail.split(stop, 1)[0]
    return tail


@needs_harness
def test_report_contains_three_endpoint_tables(dryrun):
    report = dryrun["report"]["path"]
    assert report.exists(), "REPORT.md was not written"
    text = report.read_text(encoding="utf-8")

    tables = {
        "E1": "### E1.a Admission and joint success per (model, envelope)",
        "E2": "### E2.a Weak-grader false accepts by seeded-bad family",
        "E3": "### E3.a Converse-guard rate per (model, variant)",
    }
    for experiment, heading in tables.items():
        body = _section_body(text, heading)
        assert "|---" in body, f"{experiment} table has no markdown separator"
        data_rows = [line for line in body.splitlines() if line.startswith("| ") and "---" not in line]
        assert len(data_rows) >= 1, f"{experiment} table has no data rows: {body!r}"

    assert "## What this does not show" in text
    assert "## Provenance" in text
    provenance = _section_body(text, "## Provenance")
    assert "git sha" in provenance
    assert "resume" in text or "Recompute" in text


@needs_harness
def test_report_numbers_are_recomputable(dryrun):
    """The report's E1 unit count must equal the records actually in the JSONL."""
    text = dryrun["report"]["path"].read_text(encoding="utf-8")
    records = read_records(dryrun["E1"]["jsonl"])
    body = _section_body(text, "### E1.a Admission and joint success per (model, envelope)")
    data_rows = [line for line in body.splitlines() if line.startswith("| ") and "---" not in line]
    header = [cell.strip() for cell in data_rows[0].strip("|").split("|")]
    assert header[0] == "model" and header[1] == "envelope"
    total_units = 0
    for line in data_rows[1:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        total_units += int(cells[2])
    assert total_units == len(records), "E1 table unit total does not match the JSONL"


def test_summarize_survives_missing_inputs(tmp_path):
    """A missing input must still produce the full report shape, not a crash."""
    report = tmp_path / "REPORT.md"
    result = subprocess.run(
        [
            sys.executable,
            str(SUMMARIZE),
            "--e1",
            str(tmp_path / "nope_e1.jsonl"),
            "--e2",
            str(tmp_path / "nope_e2.jsonl"),
            "--e3",
            str(tmp_path / "nope_e3.jsonl"),
            "--out",
            str(report),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=clean_env(),
    )
    assert result.returncode == 0, result.stderr
    text = report.read_text(encoding="utf-8")
    for heading in ("### E1.a", "### E2.a", "### E3.a", "## What this does not show", "## Provenance"):
        assert heading in text
    assert "not found" in text
