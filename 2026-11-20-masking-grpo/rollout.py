"""Prompt construction, sampling, admission and reward for one rollout group.

The corpus, the admission consumer and the graders are **reused, not
reimplemented**: this module imports ``harness.envelopes.admit_strict``,
``harness.graders.grade_obligation`` and ``harness.tasks_io.*`` from the sibling
``2026-11-20-frontier-reexperiment`` experiment, which is the corpus of record
(``docs/SPEC.md``, "Tasks" and "Reward and admission").

Design notes
------------
* **The model is injected.**  :func:`run_group` takes a ``generate_fn`` and never
  imports torch, so the unit tests exercise the whole sampling -> admission ->
  grading -> reward path with canned completions and no GPU or network.
* **The prompt is a strict-JSON contract.**  It states that the reply must be one
  JSON object with a single string field ``expression`` and nothing else, and it
  never shows the model the reference fix, the alternate fixes or the seeded-bad
  expressions.  :func:`build_prompt` only reads ``id``/``family``/``editable``/
  ``property``/``placeholder``/``source``; a test asserts that the answer-key
  fields cannot leak.
* **Grading is deterministic, so it is cached.**  ``execute_patch`` runs each case
  in a forked/spawned child process; a temperature-1.0 policy repeats expressions
  often, so a per-process cache keyed on ``(task_id, expression)`` removes most of
  that cost.  The cache is bounded; a cache miss changes nothing but the clock.
* **Records are plain dicts.**  One record per attempted completion, failures
  included, with the completion bytes kept so an admission decision can be
  re-litigated without resampling.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

# --- reuse the corpus of record ------------------------------------------------

#: ``.../experiments/2026-11-20-frontier-reexperiment``; it owns ``harness/`` and
#: ``tasks/``.  Resolved from this file, so the module works from any cwd.
FRONTIER_ROOT: Path = Path(__file__).resolve().parents[1] / "2026-11-20-frontier-reexperiment"
if str(FRONTIER_ROOT) not in sys.path:
    sys.path.insert(0, str(FRONTIER_ROOT))

import grpo  # noqa: E402  (stdlib-only; owns the arm enum and invalid_reward)
from harness.envelopes import admit_strict  # noqa: E402
from harness.graders import HOLDS, UNKNOWN, VIOLATED, grade_obligation  # noqa: E402
from harness.tasks_io import execute_patch, load_task  # noqa: E402

DEFAULT_TASKS_DIR: Path = FRONTIER_ROOT / "tasks"

# --- decoding, frozen by docs/SPEC.md -----------------------------------------

TEMPERATURE: float = 1.0
TOP_P: float = 0.95
MAX_NEW_TOKENS: int = 48
G: int = 8

#: Train on the 8 lexicographically first tasks, hold out the last 4.  The split
#: is computed from filenames, never hardcoded, so a corpus edit is visible.
TRAIN_TASKS: int = 8
HELD_OUT_TASKS: int = 4

GenerateFn = Callable[[str, int, int], Sequence["Generation"]]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def build_prompt(task: dict) -> str:
    """The strict-JSON repair prompt for one task.

    Every string is derived from the task file.  The answer-key fields
    (``reference_fix``, ``alternate_fixes``, ``seeded_bad``) are never read.
    """

    task_id = task["id"]
    family = task.get("family", "unknown")
    editable = task.get("editable") or {}
    path = editable.get("path", "the editable region")
    placeholder = task["placeholder"]
    property_text = " ".join(str(task["property"]).split())
    source = task["source"]

    return (
        "You are repairing one security-relevant Python predicate.\n"
        "\n"
        f"Task: {task_id}\n"
        f"Family: {family}\n"
        f"Editable region: {path}\n"
        "\n"
        "What the repaired code must guarantee:\n"
        f"{property_text}\n"
        "\n"
        f"The module source follows. The token {placeholder} marks the only region you may\n"
        "replace.\n"
        "\n"
        "--- source begin ---\n"
        f"{source}"
        "--- source end ---\n"
        "\n"
        f"Reply with the replacement for {placeholder} only.\n"
        "\n"
        "Output contract: reply with exactly one JSON object and nothing else. The object\n"
        'must have exactly one field, named "expression", whose value is a string holding\n'
        "the replacement Python expression. Do not add prose, comments, a markdown fence,\n"
        "or any other field: one JSON object with a single string field `expression`,\n"
        "nothing else.\n"
        "\n"
        "Example of a valid reply (the expression is illustrative, not the answer):\n"
        '{"expression": "user.tenant_id == resource.tenant_id"}\n'
        "\n"
        "Begin your reply with { and end it with }. Keep the whole reply under 60 tokens.\n"
    )


# ---------------------------------------------------------------------------
# Generations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Generation:
    """One sampled completion plus its token accounting."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _coerce_generation(raw: object) -> Generation:
    if isinstance(raw, Generation):
        return raw
    if isinstance(raw, str):
        return Generation(text=raw)
    if isinstance(raw, dict):
        text = raw.get("text", raw.get("completion"))
        if not isinstance(text, str):
            raise TypeError(f"generation dict has no string 'text': {raw!r}")
        return Generation(
            text=text,
            prompt_tokens=int(raw.get("prompt_tokens") or 0),
            completion_tokens=int(raw.get("completion_tokens") or 0),
        )
    raise TypeError(f"unsupported generation type {type(raw).__name__}")


def group_seed(seed: int, task_id: str, group_index: int) -> int:
    """Deterministic per-group sampling seed (stable across processes)."""

    digest = hashlib.sha256(f"{int(seed)}|{task_id}|{int(group_index)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


# ---------------------------------------------------------------------------
# Admission + reward
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Score:
    """The verdict for one completion."""

    admitted: bool
    admission_reason: str
    expression: Optional[str]
    verdict: str
    reasons: tuple[str, ...]
    reward: float


class GraderCache:
    """Bounded ``(task_id, expression) -> (verdict, reasons)`` memo.

    Grading is a pure function of the task and the expression, so a repeat is
    served from memory.  The bound keeps a long run from growing without limit.
    """

    def __init__(self, max_entries: int = 4096) -> None:
        self.max_entries = int(max_entries)
        self._entries: dict[tuple[str, str], tuple[str, tuple[str, ...]]] = {}
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, task_id: str, expression: str):
        return self._entries.get((task_id, expression))

    def put(self, task_id: str, expression: str, verdict: str, reasons: Sequence[str]) -> None:
        if len(self._entries) >= self.max_entries:
            self._entries.clear()
        self._entries[(task_id, expression)] = (verdict, tuple(reasons))


def score_completion(
    task: dict,
    completion: str,
    cache: Optional[GraderCache] = None,
    arm: str = "control",
) -> Score:
    """Admit, execute and grade one completion; reward depends on the arm.

    The admission consumer is ``admit_strict`` on the raw completion.  Only an
    admitted completion is rendered and executed; a rejected one yields
    ``verdict=UNKNOWN`` with the grader's own ``NO_EXPRESSION`` reason, which is
    exactly how ``grade_obligation`` reports "nothing to grade".

    Reward (``docs/SPEC.md``, "Reward and admission" + "Amendment v2"):

    * admitted and HOLDS -> ``1.0``;
    * admitted and not HOLDS -> ``0.0`` (a wrong but valid answer);
    * rejected (not admitted) -> ``grpo.invalid_reward(arm)``: ``0.0`` for
      ``control`` and ``masked``, ``-1.0`` for ``penalty``.

    ``admitted`` stays ``False`` and ``verdict`` stays ``"UNKNOWN"`` for a
    rejected completion in every arm: those fields describe what happened, while
    ``reward`` is the arm's valuation of it.
    """

    if not isinstance(completion, str):
        raise TypeError(f"completion must be str, not {type(completion).__name__}")

    invalid = grpo.invalid_reward(arm)  # validates the arm even when nothing is rejected

    admission = admit_strict(completion)
    expression = admission.expression if admission.ok else None

    task_id = str(task.get("id"))
    cached = cache.get(task_id, expression) if (cache is not None and expression is not None) else None
    if cached is not None:
        cache.hits += 1
        verdict, reasons = cached
    else:
        result = execute_patch(task, expression)
        verdict, reason_list = grade_obligation(task, result, expression)
        reasons = tuple(str(reason) for reason in reason_list)
        if cache is not None and expression is not None:
            cache.misses += 1
            cache.put(task_id, expression, verdict, reasons)

    if not admission.ok:
        reward = invalid
    else:
        reward = 1.0 if verdict == HOLDS else 0.0
    return Score(
        admitted=bool(admission.ok),
        admission_reason=str(admission.reason),
        expression=expression,
        verdict=str(verdict),
        reasons=reasons,
        reward=reward,
    )


# ---------------------------------------------------------------------------
# One group
# ---------------------------------------------------------------------------


@dataclass
class GroupRollout:
    """Everything one group produced, with each score already valued under ``arm``.

    The scores carry the arm-dependent reward (``grpo.invalid_reward``); the
    group-relative baseline itself is applied later by :mod:`grpo`.
    """

    task_id: str
    group_index: int
    group_id: str
    arm: str
    prompt: str
    generations: list[Generation]
    scores: list[Score]
    seed: int
    records: list[dict] = field(default_factory=list)

    @property
    def completions(self) -> list[str]:
        return [generation.text for generation in self.generations]

    @property
    def rewards(self) -> list[float]:
        return [score.reward for score in self.scores]

    @property
    def valid_mask(self) -> list[bool]:
        return [score.admitted for score in self.scores]

    @property
    def prompt_tokens(self) -> int:
        return self.generations[0].prompt_tokens if self.generations else 0

    @property
    def completion_tokens(self) -> int:
        return sum(generation.completion_tokens for generation in self.generations)


def run_group(
    task: dict,
    generate_fn: GenerateFn,
    *,
    group_index: int,
    g: int = G,
    seed: int = 0,
    arm: str = "control",
    cache: Optional[GraderCache] = None,
) -> GroupRollout:
    """Sample ``g`` completions for ``task`` and score every one of them.

    ``generate_fn(prompt, n, seed)`` is injected: the real one wraps the policy
    model (``model_utils``), the dry-run one draws from canned completions and
    the tests pass their own.  It must return exactly ``n`` items, each a
    :class:`Generation`, a plain string, or a dict with a ``text`` key.
    """

    if g < 1:
        raise ValueError(f"g must be >= 1, got {g}")

    prompt = build_prompt(task)
    sample_seed = group_seed(seed, str(task["id"]), group_index)

    raw = generate_fn(prompt, g, sample_seed)
    if raw is None:
        raise ValueError("generate_fn returned None")
    generations = [_coerce_generation(item) for item in raw]
    if len(generations) != g:
        raise ValueError(f"generate_fn returned {len(generations)} completions, expected {g}")

    scores = [score_completion(task, generation.text, cache, arm=arm) for generation in generations]
    group_id = f"g{group_index:04d}"

    records = [
        {
            "task_id": str(task["id"]),
            "group_id": group_id,
            "group_index": int(group_index),
            "sample_index": index,
            "arm": arm,
            "completion": generation.text,
            "admitted": score.admitted,
            "admission_reason": score.admission_reason,
            "expression": score.expression,
            "verdict": score.verdict,
            "reasons": list(score.reasons),
            "reward": score.reward,
            "prompt_tokens": int(generation.prompt_tokens),
            "completion_tokens": int(generation.completion_tokens),
        }
        for index, (generation, score) in enumerate(zip(generations, scores))
    ]

    return GroupRollout(
        task_id=str(task["id"]),
        group_index=int(group_index),
        group_id=group_id,
        arm=arm,
        prompt=prompt,
        generations=generations,
        scores=scores,
        seed=sample_seed,
        records=records,
    )


# ---------------------------------------------------------------------------
# Corpus and the lexicographic split
# ---------------------------------------------------------------------------


class CorpusError(RuntimeError):
    """The corpus on disk does not match the preregistered 12 / 8 / 4 shape."""


def _loader_root(tasks_dir: Path) -> Path:
    """The experiment root that ``harness.tasks_io.load_task`` appends ``tasks/`` to.

    ``load_task(task_id, root)`` reads ``<root>/tasks/<task_id>.json``, so the
    directory this module is given (which *is* the tasks directory) maps to its
    parent.  That is the harness's own layout and holds for the corpus of record
    (``.../2026-11-20-frontier-reexperiment/tasks``) and for a temporary corpus
    laid out as ``<tmp>/tasks/*.json`` in the tests.
    """

    return tasks_dir.parent


def load_corpus(tasks_dir: Path | str = DEFAULT_TASKS_DIR) -> list[dict]:
    """Load every ``*.json`` in ``tasks_dir``, sorted lexicographically by filename."""

    root = Path(tasks_dir)
    if not root.is_dir():
        raise CorpusError(f"task directory not found: {root}")
    paths = sorted(root.glob("*.json"))
    if not paths:
        raise CorpusError(f"no task files under {root}")
    loader_root = _loader_root(root)
    return [load_task(path.stem, loader_root) for path in paths]


def load_task_by_id(task_id: str, tasks_dir: Path | str = DEFAULT_TASKS_DIR) -> dict:
    """Load one validated corpus task by id."""

    return load_task(str(task_id), _loader_root(Path(tasks_dir)))


def split_train_heldout(
    tasks_dir: Path | str = DEFAULT_TASKS_DIR,
) -> tuple[list[dict], list[dict]]:
    """Return ``(train, held_out)``: the 8 first tasks and the last 4.

    The split is a function of the lexicographic filename order fixed by
    ``docs/SPEC.md``.  A corpus that is not exactly 12 tasks is a protocol
    deviation and raises instead of silently changing the split.
    """

    tasks = load_corpus(tasks_dir)
    if len(tasks) != TRAIN_TASKS + HELD_OUT_TASKS:
        raise CorpusError(
            f"the preregistered corpus is {TRAIN_TASKS + HELD_OUT_TASKS} tasks "
            f"({TRAIN_TASKS} train / {HELD_OUT_TASKS} held out); found {len(tasks)} under "
            f"{tasks_dir}. A corpus change is a protocol deviation: report it in docs/SPEC.md "
            "before running."
        )
    return tasks[:TRAIN_TASKS], tasks[TRAIN_TASKS:]


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def git_sha(start: Optional[Path] = None) -> str:
    """The repository's ``HEAD`` SHA, or ``"unknown"`` outside git.

    Never raises: provenance must not be able to take down a run or a report.
    """

    directory = Path(start or __file__).resolve()
    for candidate in (directory, *directory.parents):
        if (candidate / ".git").exists():
            try:
                completed = subprocess.run(
                    ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                return "unknown"
            sha = completed.stdout.strip()
            return sha if completed.returncode == 0 and sha else "unknown"
    return "unknown"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def spec_sha256(path: Path | str) -> str:
    """SHA-256 of ``docs/SPEC.md``, used to detect a post-hoc protocol edit."""

    try:
        return sha256_text(Path(path).read_text(encoding="utf-8"))
    except OSError:
        return "unknown"


# ---------------------------------------------------------------------------
# Dry-run canned completions
# ---------------------------------------------------------------------------


def dry_run_pool(task: dict) -> list[str]:
    """A deterministic pool of canned completions for ``--dry-run``.

    The pool mixes admitted-correct, admitted-wrong, admitted-unparseable and
    rejected replies so the dry run exercises every branch: admission rate,
    HOLDS/VIOLATED/UNKNOWN, and every arm's zero-variance and skipped-group cases.
    It is derived from the task file only, so it is identical on every host and
    is *not* evidence about any model.
    """

    reference = str(task["reference_fix"])
    alternates = [str(item) for item in (task.get("alternate_fixes") or [])]
    seeded = [str(entry.get("expression")) for entry in (task.get("seeded_bad") or [])]

    pool: list[str] = [json.dumps({"expression": reference}, sort_keys=True)]
    if alternates:
        pool.append(json.dumps({"expression": alternates[0]}, sort_keys=True))
    if seeded:
        pool.append(json.dumps({"expression": seeded[0]}, sort_keys=True))
    # Admitted, executes to a syntax error -> UNKNOWN.
    pool.append(json.dumps({"expression": "(" + reference[:12] + ")"}, sort_keys=True))
    # Rejected: prose around the object.
    pool.append('Here is the fix: {"expression": "' + reference[:20] + '"}')
    # Rejected: a markdown fence is not allowed by the strict envelope.
    pool.append('```json\n{"expression": "' + reference[:20] + '"}\n```')
    # Rejected: an extra field.
    pool.append(json.dumps({"expression": reference, "confidence": 0.9}, sort_keys=True))
    return pool


def dry_run_pools(tasks: Iterable[dict]) -> dict[str, list[str]]:
    """``{prompt: canned completions}`` for every task, for the dry-run model."""

    return {build_prompt(task): dry_run_pool(task) for task in tasks}


def estimated_tokens(text: str) -> int:
    """A crude, deterministic token estimate for the dry-run path only.

    The dry run has no tokenizer, so token accounting there is an estimate and is
    labelled as one in the metrics (``token_accounting: "estimated"``).  It is
    never used on a real run.
    """

    if not text:
        return 0
    return int(math.ceil(len(text.split()) + len(text) / 8.0))


__all__ = [
    "G",
    "Generation",
    "GenerateFn",
    "GraderCache",
    "HOLDS",
    "MAX_NEW_TOKENS",
    "Score",
    "TEMPERATURE",
    "TOP_P",
    "UNKNOWN",
    "VIOLATED",
    "CorpusError",
    "DEFAULT_TASKS_DIR",
    "FRONTIER_ROOT",
    "build_prompt",
    "dry_run_pool",
    "dry_run_pools",
    "estimated_tokens",
    "git_sha",
    "group_seed",
    "load_corpus",
    "load_task_by_id",
    "run_group",
    "score_completion",
    "sha256_text",
    "spec_sha256",
    "split_train_heldout",
]
