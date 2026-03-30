"""
AST construction for a time-series processing DSL.

The goal is to generate program trees (ASTs) whose evaluation output has a
constant length, so time series of different original lengths can be
compared by their shapes.

Implementation strategy
-----------------------
1. The AST always begins by resampling the input series to `target_length`.
2. After that, we prefer length-preserving primitives in the generated tree.
3. As a safety net, `evaluate_ast` will resample to `target_length` again if
   any selected primitive changes length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

from .components import TS_BINARY_PRIMITIVES, TS_UNARY_PRIMITIVES, resample_ts


AstNodeType = Literal["input", "unary", "binary"]


@dataclass(frozen=True)
class AstNode:
    node_type: AstNodeType
    # For unary: op_name + child
    op_name: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    child: Optional["AstNode"] = None
    # For binary: op_name + left/right
    left: Optional["AstNode"] = None
    right: Optional["AstNode"] = None


@dataclass(frozen=True)
class TsAstProgram:
    root: AstNode
    target_length: int


def _make_input_node() -> AstNode:
    return AstNode(node_type="input")


def _make_unary_node(op_name: str, params: Dict[str, Any], child: AstNode) -> AstNode:
    return AstNode(
        node_type="unary",
        op_name=op_name,
        params=params,
        child=child,
    )


def _make_binary_node(op_name: str, params: Dict[str, Any], left: AstNode, right: AstNode) -> AstNode:
    return AstNode(
        node_type="binary",
        op_name=op_name,
        params=params,
        left=left,
        right=right,
    )


def _sample_window(rng: np.random.Generator, target_length: int) -> int:
    # Prefer small odd-ish windows, but allow up to ~10% of target length.
    cap = max(3, int(round(target_length * 0.1)))
    candidates = [3, 5, 7, 9, 15, 21, 31, 41, 51]
    candidates = [c for c in candidates if c <= cap]
    if not candidates:
        candidates = [cap]
    return int(rng.choice(candidates))


def _sample_int_shift(rng: np.random.Generator, target_length: int) -> int:
    # Limit shift magnitude to ~10% of the target length.
    max_shift = max(1, int(round(target_length * 0.1)))
    shift = int(rng.integers(-max_shift, max_shift + 1))
    return shift


def _sample_unary_params(
    op_name: str, rng: np.random.Generator, target_length: int
) -> Dict[str, Any]:
    # Only override parameters when the primitive supports it.
    # (Some primitives have safe defaults and we omit kwargs.)
    if op_name in {"shift", "lag"}:
        return {"shift": _sample_int_shift(rng, target_length), "pad_value": 0.0}

    if op_name in {"diff"}:
        order = int(rng.integers(1, 4))  # 1..3
        return {"order": order, "pad_value": 0.0}

    if op_name in {"rolling_mean", "rolling_std", "rolling_median", "rolling_min", "rolling_max"}:
        return {"window": _sample_window(rng, target_length)}

    if op_name in {"ema"}:
        # span roughly correlates to smoothing strength
        span = int(rng.choice([2, 3, 5, 10, 20, max(2, target_length // 10)]))
        span = max(1, span)
        return {"span": span}

    if op_name in {"gaussian_smooth"}:
        window = _sample_window(rng, target_length)
        sigma = float(max(1e-3, window / 6.0 * rng.uniform(0.5, 1.5)))
        return {"window": window, "sigma": sigma}

    if op_name in {"rolling_zscore"}:
        window = _sample_window(rng, target_length)
        return {"window": window, "eps": 1e-12}

    if op_name in {"detrend"}:
        return {"window": _sample_window(rng, target_length)}

    if op_name in {"clip"}:
        a = float(rng.choice([0.5, 1.0, 2.0, 5.0]))
        return {"lo": -a, "hi": a}

    if op_name in {"exp_clipped"}:
        clip = float(rng.choice([5.0, 10.0, 20.0, 60.0]))
        return {"clip": clip}

    if op_name in {"shrink"}:
        # shrink factor > 1 -> fewer effective samples, then resample back
        factor = float(rng.uniform(0.6, 2.0))
        return {"factor": factor}

    if op_name in {"stretch"}:
        factor = float(rng.uniform(0.6, 2.0))
        return {"factor": factor}

    if op_name == "resample":
        return {"out_len": int(target_length)}

    # Default: no extra params.
    return {}


def _choose_default_unary_primitives(
    *, allow_resample: bool, allow_variable_length_ops: bool
) -> list[str]:
    # These change length and are intentionally disabled unless requested.
    variable_length_unary = {"fft_mag", "fft_phase", "paa", "sax"}
    ops = set(TS_UNARY_PRIMITIVES.keys())

    if not allow_variable_length_ops:
        ops -= variable_length_unary

    if not allow_resample:
        ops.discard("resample")

    # Keep a stable ordering for reproducibility if needed.
    return sorted(ops)


def _choose_default_binary_primitives(*, allow_concat: bool) -> list[str]:
    ops = set(TS_BINARY_PRIMITIVES.keys())
    if not allow_concat:
        ops.discard("concat")
    return sorted(ops)


def create_ast(
    rng: Optional[np.random.Generator] = None,
    *,
    max_depth: int = 4,
    target_length: int = 256,
    p_unary: float = 0.65,
    unary_ops: Optional[list[str]] = None,
    binary_ops: Optional[list[str]] = None,
    allow_variable_length_ops: bool = False,
    allow_concat: bool = False,
    allow_resample_in_tree: bool = False,
) -> TsAstProgram:
    """
    Create a random AST program.

    The AST is guaranteed to output a time series of length `target_length`
    after evaluation.
    """
    if rng is None:
        rng = np.random.default_rng()

    target_length = int(target_length)
    if target_length <= 0:
        raise ValueError("`target_length` must be a positive integer.")
    max_depth = int(max_depth)
    if max_depth < 0:
        raise ValueError("`max_depth` must be >= 0.")

    if unary_ops is None:
        unary_ops = _choose_default_unary_primitives(
            allow_resample=allow_resample_in_tree, allow_variable_length_ops=allow_variable_length_ops
        )
    if binary_ops is None:
        binary_ops = _choose_default_binary_primitives(allow_concat=allow_concat)

    if not unary_ops:
        raise ValueError("No unary primitives available for AST generation.")
    if not binary_ops:
        # We can still build a unary-only tree, so don't force binary ops.
        binary_ops = []

    # Base: always resample the input to fixed length.
    input_node = _make_input_node()
    base = _make_unary_node("resample", {"out_len": target_length}, input_node)

    def build(d: int) -> AstNode:
        if d <= 0:
            return base

        # Decide unary vs binary.
        use_unary = (rng.random() < p_unary) or (not binary_ops)
        if use_unary:
            op_name = str(rng.choice(unary_ops))
            # Avoid resample in the tree unless explicitly allowed.
            params = _sample_unary_params(op_name, rng, target_length)
            # Some primitives have no params; _sample_unary_params returns {}.
            return _make_unary_node(op_name, params, build(d - 1))

        # Binary case.
        op_name = str(rng.choice(binary_ops))
        # Binary primitives in this module don't take extra params.
        # (alignment/padding is handled inside the primitive implementation)
        params: Dict[str, Any] = {}
        # Use smaller remaining depth for each side to control size.
        left = build(d - 1)
        right = build(d - 1)
        return _make_binary_node(op_name, params, left, right)

    root = build(max_depth)
    return TsAstProgram(root=root, target_length=target_length)


def evaluate_ast(program: TsAstProgram, x: np.ndarray) -> np.ndarray:
    """
    Evaluate `program` on input series `x`.

    Output length is always forced to `program.target_length`.
    """

    def eval_node(node: AstNode) -> np.ndarray:
        if node.node_type == "input":
            arr = np.asarray(x, dtype=np.float64)
            if arr.ndim != 1:
                arr = arr.reshape(-1)
            return arr

        if node.node_type == "unary":
            assert node.op_name is not None
            fn = TS_UNARY_PRIMITIVES[node.op_name]
            child_val = eval_node(node.child)  # type: ignore[arg-type]
            kwargs = node.params or {}
            return fn(child_val, **kwargs)

        if node.node_type == "binary":
            assert node.op_name is not None
            fn = TS_BINARY_PRIMITIVES[node.op_name]
            left_val = eval_node(node.left)  # type: ignore[arg-type]
            right_val = eval_node(node.right)  # type: ignore[arg-type]
            kwargs = node.params or {}
            return fn(left_val, right_val, **kwargs)

        raise ValueError(f"Unknown node type: {node.node_type}")

    out = eval_node(program.root)
    if len(out) != program.target_length:
        out = resample_ts(out, program.target_length)
    return out


__all__ = ["AstNode", "TsAstProgram", "create_ast", "evaluate_ast"]
