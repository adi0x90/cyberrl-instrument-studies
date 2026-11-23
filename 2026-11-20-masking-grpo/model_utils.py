"""Model and tokenizer loading, LoRA attachment, generation, adapter save/load.

Everything that needs torch/transformers/peft is imported **inside** a function.
The module therefore imports cleanly on a laptop with no GPU stack, which is what
lets ``--dry-run`` and the unit tests run offline, and it keeps the harness's
``execute_patch`` child processes (spawned from the training script) cheap,
because they re-import the training module but never touch torch.

Two backends implement the same three-method interface:

``TorchPolicy``
    The real thing: ``Qwen/Qwen2.5-Coder-1.5B-Instruct`` + LoRA (r=16, q/v),
    bf16/fp16, temperature 1.0 / top_p 0.95 / 96 new tokens, AdamW lr 1e-5,
    one optimizer step per effective group.

``DryRunModel``
    Deterministic canned completions derived from the task file.  No torch, no
    GPU, no network, no adapter.  It exists to prove the plumbing and to let the
    report build end to end on a laptop.  **Dry-run artefacts are not evidence
    about any model.**

The model revision is recorded twice: what was *requested* (``--model-revision``
or ``MASKING_GRPO_MODEL_REVISION``) and what the loader actually *resolved*
(``config._commit_hash``, set by ``transformers`` from the downloaded snapshot).
That way the manifest names the real commit even when the requested revision is a
floating branch — and the run is marked ``pinned: false`` so the deviation is
visible rather than silent.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import grpo
import rollout
from rollout import Generation

MODEL_ID: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

#: Environment variable an operator can use instead of ``--model-revision``.
REVISION_ENV: str = "MASKING_GRPO_MODEL_REVISION"

#: Sentinels recorded when no revision was pinned / no model was loaded.
UNPINNED: str = "unpinned"
DRYRUN: str = "dry-run"

LORA_R: int = 16
LORA_ALPHA: int = 32
LORA_DROPOUT: float = 0.0
LORA_TARGETS: tuple[str, ...] = ("q_proj", "v_proj")

LEARNING_RATE: float = 1e-5
WEIGHT_DECAY: float = 0.0
DTYPE: str = "auto"


@dataclass(frozen=True)
class UpdateResult:
    """What one optimizer step (or its dry-run stand-in) returned."""

    loss: float
    sum_logprobs: list[float]
    gpu_mem_mb: Optional[float]
    stepped: bool = True


class ModelError(RuntimeError):
    """A model could not be loaded or used."""


def requested_revision(cli_value: Optional[str] = None) -> tuple[str, bool]:
    """Return ``(revision, pinned)`` from the CLI, the environment, or a sentinel."""

    candidate = cli_value or os.environ.get(REVISION_ENV) or ""
    candidate = candidate.strip()
    if candidate:
        return candidate, True
    return UNPINNED, False


def resolve_dtype(name: str = DTYPE, device: Optional[str] = None) -> str:
    """Map ``auto|bfloat16|float16|float32`` to a dtype name the loader understands."""

    name = (name or "auto").lower()
    if name in ("bfloat16", "bf16"):
        return "bfloat16"
    if name in ("float16", "fp16", "half"):
        return "float16"
    if name in ("float32", "fp32", "full"):
        return "float32"
    if name != "auto":
        raise ValueError(f"unknown dtype {name!r}; expected auto|bfloat16|float16|float32")
    try:
        import torch
    except ImportError:  # pragma: no cover - dry-run hosts have no torch
        return "float32"
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and torch.cuda.is_bf16_supported():
        return "bfloat16"
    if device.startswith("cuda"):
        return "float16"
    return "float32"


def gpu_info() -> dict:
    """``{"gpu", "gpu_memory_mb", "cuda", "device_count"}`` or nulls without a GPU."""

    info: dict = {"gpu": None, "gpu_memory_mb": None, "cuda": None, "device_count": 0}
    try:
        import torch
    except ImportError:
        return info
    try:
        info["cuda"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            info["gpu"] = str(properties.name)
            info["gpu_memory_mb"] = int(round(properties.total_memory / (1024 * 1024)))
            info["device_count"] = int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001 - provenance must never take down a run
        pass
    return info


# ---------------------------------------------------------------------------
# The real backend
# ---------------------------------------------------------------------------


class TorchPolicy:
    """``Qwen2.5-Coder-1.5B-Instruct`` + LoRA, with one-step-per-group updates."""

    def __init__(self, model, tokenizer, *, model_id: str, requested: str, resolved: str,
                 pinned: bool, dtype: str, device: str, peft=None,
                 local_files_only: bool = False) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.requested_revision = requested
        self.resolved_revision = resolved
        self.pinned = bool(pinned)
        self.dtype = dtype
        self.device = device
        self._peft = peft
        self.local_files_only = bool(local_files_only)
        self.steps = 0
        self.optimizer = None

    # -- construction ------------------------------------------------------

    @classmethod
    def load(
        cls,
        model_id: str = MODEL_ID,
        revision: Optional[str] = None,
        *,
        dtype: str = DTYPE,
        device: Optional[str] = None,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        lora_r: int = LORA_R,
        lora_alpha: int = LORA_ALPHA,
        lora_dropout: float = LORA_DROPOUT,
        lora_targets: Sequence[str] = LORA_TARGETS,
        adapter_path: Optional[str] = None,
        local_files_only: bool = False,
    ) -> "TorchPolicy":
        """Load the policy and attach (or restore) LoRA.

        ``local_files_only=True`` forbids the Hugging Face Hub: the model must
        already be in the local cache.  That is the switch this harness exposes so
        a run can be proven to make no network call at all.
        """

        try:
            import torch
            from peft import LoraConfig, PeftModel, get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - GPU host only
            raise ModelError(
                "torch/transformers/peft are not installed; install requirements.txt on the "
                f"GPU host, or use --dry-run ({exc})"
            ) from exc

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype_name = resolve_dtype(dtype, device)
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype_name]

        requested, pinned = requested_revision(revision)
        load_revision = None if requested in (UNPINNED, DRYRUN) else requested

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, revision=load_revision, local_files_only=local_files_only
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=load_revision,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                local_files_only=local_files_only,
            )
        except Exception as exc:  # noqa: BLE001 - a load failure must be one clear error
            raise ModelError(
                f"could not load {model_id} (revision={load_revision!r}, "
                f"local_files_only={local_files_only}): {type(exc).__name__}: {exc}. "
                "If the model is not cached yet, download it once on the GPU host or drop "
                "--local-files-only; a real training step is never attempted before this "
                "succeeds."
            ) from exc

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model.to(device)

        resolved = str(getattr(getattr(model, "config", None), "_commit_hash", "") or "")
        if not resolved:
            # Not all loaders populate _commit_hash; fall back to the requested tag.
            resolved = requested

        if adapter_path:
            try:
                model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
            except Exception as exc:  # noqa: BLE001
                raise ModelError(
                    f"could not load the LoRA adapter at {adapter_path}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
        else:
            lora = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=list(lora_targets),
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora)

        model.train()
        policy = cls(
            model,
            tokenizer,
            model_id=model_id,
            requested=requested,
            resolved=resolved,
            pinned=pinned,
            dtype=dtype_name,
            device=device,
            peft=(PeftModel, get_peft_model),
            local_files_only=bool(local_files_only),
        )
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        policy.optimizer = torch.optim.AdamW(
            trainable, lr=learning_rate, weight_decay=weight_decay
        )
        return policy

    # -- description -------------------------------------------------------

    def description(self) -> dict:
        return {
            "kind": "torch",
            "model_id": self.model_id,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "pinned": self.pinned,
            "dtype": self.dtype,
            "device": self.device,
            "local_files_only": self.local_files_only,
            "lora": {
                "r": LORA_R,
                "alpha": LORA_ALPHA,
                "dropout": LORA_DROPOUT,
                "target_modules": list(LORA_TARGETS),
            },
            "optimizer": {
                "name": "AdamW",
                "lr": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
            },
            **gpu_info(),
        }

    # -- generation --------------------------------------------------------

    def generate(self, prompt: str, n: int, seed: int) -> list[Generation]:
        """Sample ``n`` completions at temperature 1.0 / top_p 0.95 / 96 new tokens."""

        import torch

        tokenizer = self.tokenizer
        messages = [{"role": "user", "content": prompt}]
        chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(chat, return_tensors="pt", add_special_tokens=False).to(self.device)
        prompt_tokens = int(encoded["input_ids"].shape[1])

        torch.manual_seed(int(seed) & 0x7FFFFFFF)
        self.model.eval()
        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                do_sample=True,
                temperature=rollout.TEMPERATURE,
                top_p=rollout.TOP_P,
                max_new_tokens=rollout.MAX_NEW_TOKENS,
                num_return_sequences=int(n),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        self.model.train()

        new_tokens = output[:, prompt_tokens:]
        texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        generations = []
        for index, text in enumerate(texts):
            completion_tokens = int((new_tokens[index] != tokenizer.pad_token_id).sum().item())
            generations.append(
                Generation(
                    text=str(text),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            )
        return generations

    # -- one gradient step -------------------------------------------------

    def update(
        self,
        prompt: str,
        completions: Sequence[str],
        advantages: Sequence[float],
        treated: Sequence[bool],
    ) -> UpdateResult:
        """One optimizer step on ``-mean(advantage * sum_logprob)``.

        The loss is computed from the tensors that are backpropagated; the scalar
        is then cross-checked against :func:`grpo.group_loss` on the detached
        per-sequence log-probabilities, so the GPU path and the pure-python path
        cannot silently disagree.
        """

        import torch

        if len(completions) != len(advantages) or len(completions) != len(treated):
            raise ValueError("completions, advantages and treated must be the same length")
        if not completions:
            return UpdateResult(loss=0.0, sum_logprobs=[], gpu_mem_mb=self.gpu_mem_mb(), stepped=False)

        tokenizer = self.tokenizer
        messages = [{"role": "user", "content": prompt}]
        chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = list(tokenizer(chat, add_special_tokens=False)["input_ids"])

        sequences: list[list[int]] = []
        labels: list[list[int]] = []
        for text in completions:
            completion_ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
            if not completion_ids:
                completion_ids = [tokenizer.eos_token_id]
            sequences.append(prompt_ids + completion_ids)
            labels.append([-100] * len(prompt_ids) + completion_ids)

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        width = max(len(sequence) for sequence in sequences)

        input_ids, attention, label_rows = [], [], []
        for sequence, label in zip(sequences, labels):
            padding = width - len(sequence)
            input_ids.append(sequence + [pad_id] * padding)
            attention.append([1] * len(sequence) + [0] * padding)
            label_rows.append(label + [-100] * padding)

        device = self.device
        input_t = torch.tensor(input_ids, dtype=torch.long, device=device)
        attention_t = torch.tensor(attention, dtype=torch.long, device=device)
        label_t = torch.tensor(label_rows, dtype=torch.long, device=device)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(input_ids=input_t, attention_mask=attention_t)
        logprobs = torch.log_softmax(output.logits.float(), dim=-1)

        # logprob of token t is taken at position t-1; only completion tokens count.
        gathered = logprobs[:, :-1, :].gather(-1, input_t[:, 1:].unsqueeze(-1)).squeeze(-1)
        completion_mask = (label_t[:, 1:] != -100).float()
        sum_logprobs = (gathered * completion_mask).sum(dim=1)

        advantage_t = torch.tensor(list(advantages), dtype=torch.float32, device=device)
        treated_t = torch.tensor([1.0 if flag else 0.0 for flag in treated], dtype=torch.float32, device=device)
        denominator = treated_t.sum().clamp(min=1.0)
        loss = -(advantage_t * sum_logprobs * treated_t).sum() / denominator

        loss.backward()
        self.optimizer.step()
        self.steps += 1

        sums_python = [float(value) for value in sum_logprobs.detach().cpu().tolist()]
        expected = grpo.group_loss(advantages, sums_python, treated)
        if abs(expected - float(loss.detach())) > 1e-3 * max(1.0, abs(expected)):
            raise ModelError(
                "loss mismatch between the torch path and grpo.group_loss: "
                f"{float(loss.detach()):.6f} vs {expected:.6f}"
            )

        return UpdateResult(
            loss=float(loss.detach()),
            sum_logprobs=sums_python,
            gpu_mem_mb=self.gpu_mem_mb(),
            stepped=True,
        )

    # -- misc --------------------------------------------------------------

    def gpu_mem_mb(self) -> Optional[float]:
        try:
            import torch

            if torch.cuda.is_available():
                return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
        except Exception:  # noqa: BLE001
            pass
        return None

    def save_adapter(self, path: Path | str) -> None:
        """Save the LoRA adapter and the tokenizer to ``path``."""

        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(target))
        self.tokenizer.save_pretrained(str(target))


def load_for_evaluation(
    model_id: str = MODEL_ID,
    revision: Optional[str] = None,
    *,
    adapter_path: Optional[str] = None,
    dtype: str = DTYPE,
    device: Optional[str] = None,
    local_files_only: bool = False,
) -> TorchPolicy:
    """Load the base policy with a saved LoRA adapter (held-out evaluation)."""

    return TorchPolicy.load(
        model_id,
        revision,
        dtype=dtype,
        device=device,
        adapter_path=adapter_path,
        local_files_only=local_files_only,
    )


# ---------------------------------------------------------------------------
# The dry-run backend
# ---------------------------------------------------------------------------


class DryRunModel:
    """Deterministic canned model: no torch, no GPU, no network, no adapter.

    ``generate`` answers from the per-prompt pool built by
    :func:`rollout.dry_run_pools`.  Roughly one group in five is forced to repeat
    a single pool entry, which is what produces the zero-variance and
    skipped-group cases the report has to render.  The draw depends on the arm, so
    the arms do not see identical completions (they will not in a real run
    either); the *prompt* is identical across arms, which is what the protocol's
    paired variance comparison keys on.
    """

    def __init__(
        self,
        pools: dict[str, list[str]],
        *,
        seed: int = 0,
        arm: str = "control",
        model_id: str = DRYRUN,
        requested_revision: str = DRYRUN,
    ) -> None:
        if not pools:
            raise ValueError("DryRunModel needs at least one prompt pool")
        self.pools = pools
        self.seed = int(seed)
        self.arm = arm
        self.model_id = model_id
        self.requested_revision = requested_revision
        self.steps = 0
        self._tag = f"dry-run|{arm}"

    # -- description -------------------------------------------------------

    def description(self) -> dict:
        return {
            "kind": "dry-run",
            "model_id": self.model_id,
            "requested_revision": self.requested_revision,
            "resolved_revision": DRYRUN,
            "pinned": False,
            "dtype": "float32",
            "device": "cpu",
            "lora": {
                "r": LORA_R,
                "alpha": LORA_ALPHA,
                "dropout": LORA_DROPOUT,
                "target_modules": list(LORA_TARGETS),
                "attached": False,
            },
            "optimizer": {
                "name": "AdamW",
                "lr": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "steps": 0,
            },
            **gpu_info(),
            "note": "dry run: canned completions, no model, no gradient. Not evidence.",
        }

    # -- sampling ----------------------------------------------------------

    def _digest(self, *parts: object) -> bytes:
        payload = "|".join(str(part) for part in (self._tag, self.seed, *parts))
        return hashlib.sha256(payload.encode("utf-8")).digest()

    def generate(self, prompt: str, n: int, seed: int) -> list[Generation]:
        pool = self.pools.get(prompt)
        if pool is None:
            raise KeyError(
                "dry-run model has no canned pool for this prompt; construct it with "
                "rollout.dry_run_pools(tasks)"
            )
        base = self._digest(prompt, seed, "group")
        degenerate = base[0] % 5 == 0
        if degenerate:
            indices = [base[1] % len(pool)] * int(n)
        else:
            indices = [
                self._digest(prompt, seed, index, "sample")[0] % len(pool) for index in range(int(n))
            ]

        prompt_tokens = rollout.estimated_tokens(prompt)
        return [
            Generation(
                text=pool[index],
                prompt_tokens=prompt_tokens,
                completion_tokens=rollout.estimated_tokens(pool[index]),
            )
            for index in indices
        ]

    # -- one "gradient step" ----------------------------------------------

    def _logprob(self, prompt: str, completion: str) -> float:
        digest = hashlib.sha256(
            f"logprob|{self.seed}|{prompt}|{completion}".encode("utf-8")
        ).digest()
        unit = int.from_bytes(digest[:4], "big") / float(0xFFFFFFFF)
        return -(0.5 + 3.0 * unit)

    def update(
        self,
        prompt: str,
        completions: Sequence[str],
        advantages: Sequence[float],
        treated: Sequence[bool],
    ) -> UpdateResult:
        sum_logprobs = [self._logprob(prompt, completion) for completion in completions]
        loss = grpo.group_loss(advantages, sum_logprobs, treated)
        self.steps += 1
        return UpdateResult(loss=float(loss), sum_logprobs=sum_logprobs, gpu_mem_mb=None, stepped=True)

    # -- misc --------------------------------------------------------------

    def gpu_mem_mb(self) -> Optional[float]:
        return None

    def save_adapter(self, path: Path | str) -> None:  # pragma: no cover - never called
        raise ModelError("a dry run has no adapter to save")


__all__ = [
    "DRYRUN",
    "DTYPE",
    "DryRunModel",
    "LEARNING_RATE",
    "LORA_ALPHA",
    "LORA_DROPOUT",
    "LORA_R",
    "LORA_TARGETS",
    "MODEL_ID",
    "ModelError",
    "REVISION_ENV",
    "TorchPolicy",
    "UNPINNED",
    "UpdateResult",
    "WEIGHT_DECAY",
    "gpu_info",
    "load_for_evaluation",
    "requested_revision",
    "resolve_dtype",
]
