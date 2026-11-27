"""Seeded input generators for the twelve expression-control tasks.

Each generator emits **inputs only** (call expressions).  It never emits an
expected answer: labels come from the differential oracle in ``grade_property``,
which is built from independent valid implementations.  That separation is what
keeps the property tier from encoding one implementation's semantics as truth.

Generators are boundary-weighted (a fixed grid of contract edges) plus random
draws, because a uniform random sample almost never lands on the edges that
separate a correct guard from a plausible one.
"""

from __future__ import annotations

import random
from typing import Callable

Rng = random.Random


# ---------------------------------------------------------------------------
# Shared value pools
# ---------------------------------------------------------------------------

_DIGITS = "0123456789"
_ASCII_PRINT = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
_UNICODE_DIGITS = ["\u0661\u0662\u0663", "\uff11\uff12\uff13", "\u00b2", "\u00b3\u00b9"]
_CONFUSABLES = ["\u0430", "\u03bf", "\u0131", "\u200b", "\u200c", "\u00ad"]  # cyr a, omicron, dotless i, ZWSP, ZWNJ, soft hyphen


def _digits(rng: Rng, max_len: int = 24) -> str:
    length = rng.choice([0, 1, 2, 3, 4, 6, 8, 12, 18, max_len])
    return "".join(rng.choice(_DIGITS) for _ in range(length))


def _text(rng: Rng, max_len: int = 16) -> str:
    length = rng.choice([0, 1, 2, 3, 5, 8, max_len])
    return "".join(rng.choice(_ASCII_PRINT) for _ in range(length))


def _ip(rng: Rng) -> str:
    return ".".join(str(rng.randrange(0, 256)) for _ in range(4))


# ---------------------------------------------------------------------------
# Per-task generators: (rng) -> call expression
# ---------------------------------------------------------------------------


def _bounds_content_length(rng: Rng) -> str:
    caps = [0, 1, 64, 1024, 4096, 65535, 1048576, 10**9]
    cap = rng.choice(caps)
    near = rng.choice([cap - 1, cap, cap + 1, cap * 2, 0, 1])
    style = rng.randrange(0, 8)
    if style == 0:
        raw = str(max(near, 0))
    elif style == 1:
        raw = "0" * rng.randrange(1, 4) + str(max(near, 0))
    elif style == 2:
        raw = _digits(rng, 30)
    elif style == 3:
        raw = rng.choice(["-1", "+1", " 1", "1 ", "1.0", "4e3", "0x10", "1_000", ""])
    elif style == 4:
        raw = rng.choice(_UNICODE_DIGITS)
    elif style == 5:
        raw = _text(rng, 12)
    elif style == 6:
        raw = str(rng.randrange(0, cap + 2))
    else:
        raw = "9" * rng.randrange(1, 26)
    return f"length_ok({raw!r}, Limits({cap}))"


def _bounds_header_budget(rng: Rng) -> str:
    max_count = rng.choice([0, 1, 2, 3, 10, 100])
    max_bytes = rng.choice([0, 1, 5, 20, 64, 1024, 8192])
    count = rng.choice([0, 1, 2, 3, max_count - 1, max_count, max_count + 1, 101])
    count = max(0, min(count, 120))
    values = []
    budget_left = max_bytes
    for _ in range(count):
        style = rng.randrange(0, 7)
        if style == 0:
            value = ""
        elif style == 1:
            value = "x" * rng.randrange(0, max(1, budget_left) + 2 if budget_left >= 0 else 2)
        elif style == 2:
            value = "ok\r\nInjected: 1"
        elif style == 3:
            value = "bad\x00value"
        elif style == 4:
            value = "a" * rng.randrange(0, 6)
        elif style == 5:
            value = _text(rng, 8)
        else:
            value = "y" * rng.randrange(0, 40)
        budget_left -= len(value)
        values.append(value)
    return f"headers_ok({values!r}, HeaderBudget({max_count}, {max_bytes}))"


def _canonicalization_identity_fold(rng: Rng) -> str:
    pool = [
        "alice", "Alice", "ALICE", "alice\u200b", "admin", "Admin ", "user",
        "user\u00a0", "caf\u00e9", "cafe\u0301", "\ufb01le", "file", "stra\u00dfe",
        "STRASSE", "bob", "BOB", "", "a" * 8, "\u212a", "k", "\u0430lice",
    ]
    first = rng.choice(pool)
    style = rng.randrange(0, 4)
    if style == 0:
        second = first
    elif style == 1:
        second = first.upper() if rng.random() < 0.5 else first.lower()
    elif style == 2:
        second = first + rng.choice(_CONFUSABLES)
    else:
        second = rng.choice(pool)
    return f"claims_account({first!r}, {second!r})"


def _canonicalization_signed_path(rng: Rng) -> str:
    segments = ["admin", "report", "secret", "..", ".", "%2e%2e", "%00.png", "sub", "", "x", "%2f"]
    length = rng.randrange(0, 5)
    parts = [rng.choice(segments) for _ in range(length)]
    signed = "/" + "/".join(parts) if rng.random() < 0.85 else "/".join(parts)
    request = signed
    style = rng.randrange(0, 6)
    if style == 0:
        request = signed + "/"
    elif style == 1:
        request = signed.replace("/", "//", 1) if "/" in signed else "/" + signed
    elif style == 2:
        request = (signed + "/").replace("//", "/./", 1) if signed else "/./"
    elif style == 3:
        request = "/" + "/".join(parts + [rng.choice(segments)])
    elif style == 4:
        request = "/" + "/".join([".."] + parts)
    else:
        request = signed
    return f"signature_valid({signed!r}, {request!r})"


def _containment_cidr_block(rng: Rng) -> str:
    base = _ip(rng)
    style = rng.randrange(0, 4)
    if style == 0:
        addr = _ip(rng)
    elif style == 1:
        # same /16, different host bits
        prefix = rng.choice([8, 16, 24])
        base_parts = base.split(".")
        keep = prefix // 8
        addr = ".".join(base_parts[:keep] + [str(rng.randrange(0, 256)) for _ in range(4 - keep)])
    elif style == 2:
        addr = base
    else:
        addr = base
    prefix = rng.choice([0, 1, 7, 8, 9, 16, 24, 31, 32, 33])
    return f"contained(Network(parse_ip({base!r}), {prefix}), Address(parse_ip({addr!r})))"


def _containment_path_prefix(rng: Rng) -> str:
    scopes = ["tenant-a", "tenant-b", "a", "tenant-a/sub", ""]
    segments = ["tenant-a", "tenant-b", "sub", "..", ".", "x", "x..y", "", "secret"]
    scope = rng.choice(scopes)
    length = rng.randrange(0, 5)
    key = "/".join(rng.choice(segments) for _ in range(length))
    if rng.random() < 0.5 and scope:
        key = scope + "/" + key if key else scope + "/"
    style = rng.randrange(0, 4)
    if style == 0 and scope:
        key = scope + "/../" + rng.choice(segments)
    elif style == 1 and scope:
        key = scope + "2/" + rng.choice(segments)
    return f"contained(Scope({scope!r}), Target({key!r}))"


def _cookie_host_prefix(rng: Rng) -> str:
    names = ["__Host-sid", "__host-sid", "__Host-", "sid", "__Secure-x", "__Host-csrf",
             "__Host-" + _text(rng, 8), ""]
    name = rng.choice(names)
    secure = rng.random() < 0.7
    path = rng.choice(["/", "/app", "", "/app/", "//"])
    domain = rng.choice([None, None, "example.com", ".example.com"])
    http_only = rng.random() < 0.7
    return f"host_prefix_ok(Cookie({name!r}, {secure!r}, {path!r}, {domain!r}, {http_only!r}))"


def _cookie_hostonly_forward(rng: Rng) -> str:
    hosts = ["app.example.com", "evil.example.com", "example.com", "sub.app.example.com",
             "APP.example.com", ""]
    cookie_host = rng.choice(hosts)
    request_host = rng.choice(hosts) if rng.random() < 0.7 else cookie_host
    cookie_secure = rng.random() < 0.6
    request_secure = rng.random() < 0.5
    return (
        f"may_attach(Cookie('sid', {cookie_host!r}, {cookie_secure!r}), "
        f"Request({request_host!r}, {request_secure!r}))"
    )


def _state_drift_quorum_generation(rng: Rng) -> str:
    owners = ["n1", "n2", "node-7", "c-1", ""]
    owner = rng.choice(owners)
    node = rng.choice(owners) if rng.random() < 0.7 else owner
    lease_gen = rng.choice([0, 1, 2, 4, 5, 12, 2**31 - 1])
    membership_gen = lease_gen if rng.random() < 0.5 else rng.choice([0, 1, 2, 3, 5, 13])
    released = rng.random() < 0.3
    return f"may_commit(Lease({owner!r}, {lease_gen}, {released!r}), Membership({node!r}, {membership_gen}))"


def _state_drift_revocation_epoch(rng: Rng) -> str:
    subjects = ["u1", "u2", "user-42", ""]
    subject = rng.choice(subjects)
    state_subject = subject if rng.random() < 0.75 else rng.choice(subjects)
    session_epoch = rng.choice([0, 1, 7, 41, 42, 43, 10**6])
    state_epoch = session_epoch if rng.random() < 0.5 else rng.choice([0, 1, 7, 41, 42, 43, 10**6])
    settled = rng.random() < 0.7
    return f"may_use(Session({subject!r}, {session_epoch}), RevocationState({state_subject!r}, {state_epoch}, {settled!r}))"


def _authz_delegated_scope(rng: Rng) -> str:
    tokens = ["orders:read", "orders:write", "orders:write:admin", "reports:read", "read", "write"]
    scopes = [rng.choice(tokens) for _ in range(rng.randrange(0, 4))]
    subject = rng.choice(["u1", "u2", "svc-a", ""])
    delegated_by = rng.choice([None, None, "svc-a", "svc-b"])
    op_scope = rng.choice(tokens)
    delegation_allowed = rng.random() < 0.5
    return (
        f"may_invoke(Principal({subject!r}, {scopes!r}, {delegated_by!r}), "
        f"Operation({op_scope!r}, {delegation_allowed!r}))"
    )


def _authz_delegated_write(rng: Rng) -> str:
    tenants = ["t1", "t2", "tenant-42", ""]
    tenant = rng.choice(tenants)
    resource_tenant = tenant if rng.random() < 0.6 else rng.choice(tenants)
    scopes = rng.choice([[], ["read"], ["write"], ["read", "write"], ["write_documents"], ["write", "write"]])
    active = rng.random() < 0.6
    agent = rng.choice(["agent-a", "agent-b", ""])
    return (
        f"may_write(Context({tenant!r}, {agent!r}, {scopes!r}, {active!r}), "
        f"Resource({resource_tenant!r}))"
    )


GENERATORS: dict[str, Callable[[Rng], str]] = {
    "authz_delegated_scope_02": _authz_delegated_scope,
    "authz_delegated_write_01": _authz_delegated_write,
    "bounds_content_length_01": _bounds_content_length,
    "bounds_header_budget_02": _bounds_header_budget,
    "canonicalization_identity_fold_02": _canonicalization_identity_fold,
    "canonicalization_signed_path_01": _canonicalization_signed_path,
    "containment_cidr_block_02": _containment_cidr_block,
    "containment_path_prefix_01": _containment_path_prefix,
    "cookie_host_prefix_01": _cookie_host_prefix,
    "cookie_hostonly_forward_02": _cookie_hostonly_forward,
    "state_drift_quorum_generation_02": _state_drift_quorum_generation,
    "state_drift_revocation_epoch_01": _state_drift_revocation_epoch,
}


def generate_calls(task_id: str, seed: int, n: int = 200) -> list[str]:
    """Deterministic property-case set for ``task_id`` under ``seed``."""

    generator = GENERATORS.get(task_id)
    if generator is None:
        raise KeyError(f"no property generator for task {task_id!r}")
    rng = Rng(seed)
    calls = [generator(rng) for _ in range(n)]
    # de-duplicate while preserving order: repeated calls inflate the case count
    # without adding discriminating power
    seen: set[str] = set()
    unique: list[str] = []
    for call in calls:
        if call not in seen:
            seen.add(call)
            unique.append(call)
    return unique


# ---------------------------------------------------------------------------
# Metamorphic pairs: transforms that must not change the verdict
# ---------------------------------------------------------------------------


def _mm_content_length(call_seed: int) -> list[tuple[str, str]]:
    rng = Rng(call_seed)
    pairs: list[tuple[str, str]] = []
    for _ in range(8):
        cap = rng.choice([0, 1, 64, 1024, 1048576])
        value = str(rng.randrange(0, cap + 2))
        pairs.append(
            (
                f"length_ok({value!r}, Limits({cap}))",
                f"length_ok({'0' * rng.randrange(1, 4) + value!r}, Limits({cap}))",
            )
        )
    return pairs


def _mm_header_budget(call_seed: int) -> list[tuple[str, str]]:
    rng = Rng(call_seed)
    pairs: list[tuple[str, str]] = []
    for _ in range(8):
        values = [_text(rng, 6) for _ in range(rng.randrange(0, 5))]
        budget = (rng.choice([0, 1, 3, 10]), rng.choice([0, 5, 20, 64]))
        shuffled = list(values)
        rng.shuffle(shuffled)
        pairs.append(
            (
                f"headers_ok({values!r}, HeaderBudget({budget[0]}, {budget[1]}))",
                f"headers_ok({shuffled!r}, HeaderBudget({budget[0]}, {budget[1]}))",
            )
        )
    return pairs


def _mm_identity_fold(call_seed: int) -> list[tuple[str, str]]:
    rng = Rng(call_seed)
    pairs: list[tuple[str, str]] = []
    pool = ["alice", "Alice", "admin", "user\u00a0", "caf\u00e9", "cafe\u0301", "stra\u00dfe", ""]
    for _ in range(8):
        a, b = rng.choice(pool), rng.choice(pool)
        pairs.append((f"claims_account({a!r}, {b!r})", f"claims_account({b!r}, {a!r})"))
    return pairs


def _mm_signed_path(call_seed: int) -> list[tuple[str, str]]:
    rng = Rng(call_seed)
    pairs: list[tuple[str, str]] = []
    for _ in range(8):
        segs = [rng.choice(["a", "b", "report", "admin"]) for _ in range(rng.randrange(1, 4))]
        canonical = "/" + "/".join(segs)
        doubled = canonical.replace("/", "//", 1)
        pairs.append((f"signature_valid({canonical!r}, {doubled!r})", f"signature_valid({canonical!r}, {canonical!r})"))
    return pairs


def _mm_cidr(call_seed: int) -> list[tuple[str, str]]:
    rng = Rng(call_seed)
    pairs: list[tuple[str, str]] = []
    for _ in range(8):
        base = _ip(rng)
        addr = _ip(rng)
        prefix = rng.choice([0, 8, 16, 24, 32])
        first = ".".join(str(int(p)) for p in base.split("."))
        zero_padded = ".".join("0" + p for p in first.split("."))
        pairs.append(
            (
                f"contained(Network(parse_ip({base!r}), {prefix}), Address(parse_ip({addr!r})))",
                f"contained(Network(parse_ip({zero_padded!r}), {prefix}), Address(parse_ip({addr!r})))",
            )
        )
    return pairs


def _mm_path_prefix(call_seed: int) -> list[tuple[str, str]]:
    rng = Rng(call_seed)
    pairs: list[tuple[str, str]] = []
    for _ in range(8):
        scope = rng.choice(["tenant-a", "tenant-b", "a"])
        key = scope + "/" + rng.choice(["x", "sub/y", "report.csv"])
        pairs.append((f"contained(Scope({scope!r}), Target({key!r}))", f"contained(Scope({scope!r}), Target({key.replace('/', '//', 1)!r}))"))
    return pairs


def _mm_host_prefix(call_seed: int) -> list[tuple[str, str]]:
    rng = Rng(call_seed)
    pairs: list[tuple[str, str]] = []
    for _ in range(8):
        name = "__Host-" + _text(rng, 6)
        args = (rng.random() < 0.7, rng.choice(["/", "/app"]), rng.choice([None, "example.com"]), rng.random() < 0.7)
        pairs.append(
            (
                f"host_prefix_ok(Cookie({name!r}, {args[0]!r}, {args[1]!r}, {args[2]!r}, {args[3]!r}))",
                f"host_prefix_ok(Cookie({name + '-2'!r}, {args[0]!r}, {args[1]!r}, {args[2]!r}, {args[3]!r}))",
            )
        )
    return pairs


def _mm_hostonly(call_seed: int) -> list[tuple[str, str]]:
    rng = Rng(call_seed)
    pairs: list[tuple[str, str]] = []
    for _ in range(8):
        host = rng.choice(["app.example.com", "example.com", "sub.app.example.com"])
        pairs.append(
            (
                f"may_attach(Cookie('sid', {host!r}, False), Request({host!r}, False))",
                f"may_attach(Cookie('sid', {host!r}, False), Request({host!r}, True))",
            )
        )
    return pairs


def _mm_quorum(call_seed: int) -> list[tuple[str, str]]:
    rng = Rng(call_seed)
    pairs: list[tuple[str, str]] = []
    for _ in range(8):
        owner = rng.choice(["n1", "n2"])
        node = rng.choice(["n1", "n2"])
        gen = rng.choice([0, 1, 4, 9])
        released = rng.random() < 0.3
        k = rng.randrange(1, 10)
        pairs.append(
            (
                f"may_commit(Lease({owner!r}, {gen}, {released!r}), Membership({node!r}, {gen}))",
                f"may_commit(Lease({owner!r}, {gen + k}, {released!r}), Membership({node!r}, {gen + k}))",
            )
        )
    return pairs


def _mm_revocation(call_seed: int) -> list[tuple[str, str]]:
    rng = Rng(call_seed)
    pairs: list[tuple[str, str]] = []
    for _ in range(8):
        subject = rng.choice(["u1", "u2"])
        session_epoch = rng.randrange(0, 10)
        state_epoch = rng.randrange(0, 10)
        settled = rng.random() < 0.7
        k = rng.randrange(1, 10)
        pairs.append(
            (
                f"may_use(Session({subject!r}, {session_epoch}), RevocationState({subject!r}, {state_epoch}, {settled!r}))",
                f"may_use(Session({subject!r}, {session_epoch + k}), RevocationState({subject!r}, {state_epoch + k}, {settled!r}))",
            )
        )
    return pairs


def _mm_scope(call_seed: int) -> list[tuple[str, str]]:
    rng = Rng(call_seed)
    pairs: list[tuple[str, str]] = []
    for _ in range(8):
        scopes = ["orders:read", "orders:write", "reports:read"][: rng.randrange(0, 4)]
        shuffled = list(scopes)
        rng.shuffle(shuffled)
        delegated = rng.choice([None, "svc-a"])
        op = rng.choice(["orders:read", "orders:write"])
        allowed = rng.random() < 0.5
        pairs.append(
            (
                f"may_invoke(Principal('u1', {scopes!r}, {delegated!r}), Operation({op!r}, {allowed!r}))",
                f"may_invoke(Principal('u1', {shuffled!r}, {delegated!r}), Operation({op!r}, {allowed!r}))",
            )
        )
    return pairs


def _mm_tenant_write(call_seed: int) -> list[tuple[str, str]]:
    rng = Rng(call_seed)
    pairs: list[tuple[str, str]] = []
    for _ in range(8):
        a, b = rng.choice(["t1", "t2"]), rng.choice(["t1", "t2"])
        scopes = rng.choice([[], ["read"], ["write"], ["write", "read"]])
        active = rng.random() < 0.6
        pairs.append(
            (
                f"may_write(Context({a!r}, 'agent-a', {scopes!r}, {active!r}), Resource({b!r}))",
                f"may_write(Context({b!r}, 'agent-a', {scopes!r}, {active!r}), Resource({a!r}))",
            )
        )
    return pairs


METAMORPHIC: dict[str, Callable[[int], list[tuple[str, str]]]] = {
    "authz_delegated_scope_02": _mm_scope,
    "authz_delegated_write_01": _mm_tenant_write,
    "bounds_content_length_01": _mm_content_length,
    "bounds_header_budget_02": _mm_header_budget,
    "canonicalization_identity_fold_02": _mm_identity_fold,
    "canonicalization_signed_path_01": _mm_signed_path,
    "containment_cidr_block_02": _mm_cidr,
    "containment_path_prefix_01": _mm_path_prefix,
    "cookie_host_prefix_01": _mm_host_prefix,
    "cookie_hostonly_forward_02": _mm_hostonly,
    "state_drift_quorum_generation_02": _mm_quorum,
    "state_drift_revocation_epoch_01": _mm_revocation,
}


__all__ = ["GENERATORS", "METAMORPHIC", "generate_calls"]
