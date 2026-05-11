from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from dataclasses import dataclass
from sympy import N, Eq, symbols

from sympy import Symbol, Expr

Dim = Symbol | Expr

global_counter = 0

INPUT_ROWS = Symbol("input_rows", integer=True, positive=True)
INPUT_COLUMNS = Symbol("input_columns", integer=True, positive=True)


@lru_cache(maxsize=256)
def _interp_grids(in_len: int, out_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cache the `np.interp` grids for resampling/alignment.

    The grids depend only on lengths, so caching avoids repeated `linspace()`.
    """
    in_len = int(in_len)
    out_len = int(out_len)
    xs_old = np.linspace(0.0, 1.0, num=in_len, dtype=np.float64)
    xs_new = np.linspace(0.0, 1.0, num=out_len, dtype=np.float64)
    return xs_old, xs_new


@dataclass(frozen=True)
class Shape:
    rows: Dim
    cols: Dim

    def equals(self, other):
        return [
            Eq(self.rows, other.rows),
            Eq(self.cols, other.cols),
        ]


class Primitive:
    def __init__(self):
        self.output_shape = None

        self.input_shapes = []
        self.children: list["Primitive"] = []

        # symbolic constraints accumulated here
        self.constraints = []

        self.params = {}

        self.constraint_solution = None

        self._get_constraint_counter()
        self.add_output_shape()

    def __call__(self, x: list) -> np.ndarray:
        raise NotImplementedError

    def _get_constraint_counter(self) -> int:
        global global_counter
        self.counter = global_counter
        global_counter += 1

    def add_output_shape(self):
        counter = self.counter
        self.output_shape = Shape(
            Symbol(f"{counter}_{self.name}_rows"),
            Symbol(f"{counter}_{self.name}_columns"),
        )

    def infer_parameters(self):
        raise NotImplementedError

    def add_child(self, child: "Primitive"):
        self.children.append(child)
        self.input_shapes.append(child.output_shape)

    def add_output_constraint(self, rows=None, columns=None):
        if rows is not None:
            self.constraints.append(Eq(self.output_shape.rows, rows))
        if columns is not None:
            self.constraints.append(Eq(self.output_shape.cols, columns))


class InputPrimitive(Primitive):
    def __init__(self):
        self.name = "input"
        super().__init__()

    def __call__(self, x):
        return x[0]

    def infer_parameters(self):
        self.constraints.extend(
            self.output_shape.equals(
                Shape(Symbol("input_rows"), Symbol("input_columns"))
            )
        )
        return self.constraints

    def fill_free_parameter(self):
        pass


class UnaryPrimitive(Primitive):
    def __init__(self):
        super().__init__()

    def __call__(self, x: list) -> np.ndarray:
        if len(x) != 1:
            raise ValueError(f"UnaryPrimitive expected 1 input, got {len(x)}.")
        return x[0]

    def infer_parameters(self):
        if len(self.input_shapes) != 1:
            raise ValueError("Unary primitive requires one shape")

    def add_child(self, child):
        super().add_child(child)
        if len(self.children) > 1:
            raise ValueError("Unary primitive requires one child")


class IdentityPrimitive(UnaryPrimitive):
    def __init__(self):
        self.name = "identity"

        super().__init__()

    def infer_parameters(self):
        super().infer_parameters()

        input_shape = self.input_shapes[0]

        constraints = self.constraints

        constraints.extend(input_shape.equals(self.output_shape))

        return constraints

    def __call__(self, x: list) -> np.ndarray:
        x = super().__call__(x)

        return x.copy()


class PadAndTruncateToFixedLength(UnaryPrimitive):
    def __init__(self):
        self.name = "pad_and_truncate"

        super().__init__()

        counter = self.counter
        self.fixed_out_len = Symbol(f"{counter}_fixed_out_len__pad_and_truncate")

    def infer_parameters(self):
        super().infer_parameters()

        constraints = self.constraints
        constraints.append(Eq(self.output_shape.rows, self.input_shapes[0].rows))
        constraints.append(Eq(self.fixed_out_len, self.output_shape.cols))

        return constraints

    def _resolve_fixed(self):
        if "fixed" in self.params:
            return self.params["fixed"]
        try:
            fixed = int(self.fixed_out_len.subs(self.constraint_solution))
        except:
            input_columns = int(INPUT_COLUMNS.subs(self.constraint_solution))
            fixed = np.random.randint(
                int(input_columns * 0.8), int(input_columns) * 1.2
            )
            self.constraints.append(Eq(self.fixed_out_len, fixed))
            self.params["fixed"] = fixed
        return fixed

    def __call__(self, x):
        x = super().__call__(x)

        n, m = x.shape

        fixed = self._resolve_fixed()
        if n > fixed:
            return x[:fixed, :]

        if n < fixed:
            out = np.zeros((fixed, m), dtype=x.dtype)
            out[:n, :] = x
            return out

        return x


UNARY_PRIMITIVES = [IdentityPrimitive]


# class ZscorePrimitive(UnaryPrimitive):
#     def __init__(self, output_shape):
#         super().__init__(output_shape)
#         self.name = "zscore"
#         self.params = {"eps": 1e-12}
#         self.input_shapes = [output_shape]

#     def __call__(self, x: list) -> np.ndarray:
#         super().__call__(x)
#         x = x[0]

#         eps = self.params["eps"]
#         return (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + eps)


# class MinMaxPrimitive(UnaryPrimitive):
#     def __init__(self, output_shape):
#         super().__init__(output_shape)
#         self.name = "minmax"
#         self.params = {"eps": 1e-12}
#         self.input_shapes = [output_shape]

#     def __call__(self, x: list) -> np.ndarray:
#         super().__call__(x)
#         x = x[0]

#         eps = self.params["eps"]
#         mn = np.min(x, axis=0)
#         mx = np.max(x, axis=0)
#         return (x - mn) / ((mx - mn) + eps)


# class L2NormalizePrimitive(UnaryPrimitive):
#     def __init__(self, output_shape):
#         super().__init__(output_shape)
#         self.name = "l2_normalize"
#         self.params = {"eps": 1e-12}
#         self.input_shapes = [output_shape]

#     def __call__(self, x: list) -> np.ndarray:
#         super().__call__(x)
#         x = x[0]

#         eps = self.params["eps"]
#         return x / (np.linalg.norm(x, axis=0) + eps)


# class AbsPrimitive(UnaryPrimitive):
#     def __init__(self, output_shape):
#         super().__init__(output_shape)
#         self.name = "abs"
#         self.params = None
#         self.input_shapes = [output_shape]

#     def __call__(self, x: list) -> np.ndarray:
#         super().__call__(x)
#         x = x[0]

#         return np.abs(x)


# class SignPrimitive(UnaryPrimitive):
#     def __init__(self, output_shape):
#         super().__init__(output_shape)
#         self.name = "sign"
#         self.params = None
#         self.input_shapes = [output_shape]

#     def __call__(self, x: list) -> np.ndarray:
#         super().__call__(x)
#         x = x[0]

#         return np.sign(x)


# class SquarePrimitive(UnaryPrimitive):
#     def __init__(self, output_shape):
#         super().__init__(output_shape)
#         self.name = "square"
#         self.params = None
#         self.input_shapes = [output_shape]

#     def __call__(self, x: list) -> np.ndarray:
#         super().__call__(x)
#         x = x[0]

#         return x * x


# class CubePrimitive(UnaryPrimitive):
#     def __init__(self, output_shape):
#         super().__init__(output_shape)
#         self.name = "cube"
#         self.params = None
#         self.input_shapes = [output_shape]

#     def __call__(self, x: list) -> np.ndarray:
#         super().__call__(x)
#         x = x[0]

#         return x * x * x


# class SqrtPosPrimitive(UnaryPrimitive):
#     def __init__(self, output_shape):
#         super().__init__(output_shape)
#         self.name = "sqrt_pos"
#         self.params = None
#         self.input_shapes = [output_shape]

#     def __call__(self, x: list) -> np.ndarray:
#         super().__call__(x)
#         x = x[0]

#         return np.sqrt(np.maximum(x, 0.0))


# class LogAbsPrimitive(UnaryPrimitive):
#     def __init__(self, output_shape):
#         super().__init__(output_shape)
#         self.name = "log_abs"
#         self.params = {"eps": 1e-12}
#         self.input_shapes = [output_shape]

#     def __call__(self, x: list) -> np.ndarray:
#         super().__call__(x)
#         x = x[0]

#         eps = self.params["eps"]
#         return np.log(np.abs(x) + eps)


# class ExpClippedPrimitive(UnaryPrimitive):
#     def __init__(self, output_shape):
#         super().__init__(output_shape)
#         self.name = "exp_clipped"
#         self.params = {"clip": 60.0}
#         self.input_shapes = [output_shape]

#     def __call__(self, x: list) -> np.ndarray:
#         super().__call__(x)
#         x = x[0]

#         clip = self.params["clip"]
#         return np.exp(np.clip(x, -clip, clip))


# class ReluPrimitive(UnaryPrimitive):
#     def __init__(self, output_shape):
#         super().__init__(output_shape)
#         self.name = "relu"
#         self.params = None
#         self.input_shapes = [output_shape]

#     def __call__(self, x: list) -> np.ndarray:
#         super().__call__(x)
#         x = x[0]

#         return np.maximum(x, 0.0)


# class TanhPrimitive(UnaryPrimitive):
#     def __init__(self, output_shape):
#         super().__init__(output_shape)
#         self.name = "tanh"
#         self.params = None
#         self.input_shapes = [output_shape]

#     def __call__(self, x: list) -> np.ndarray:
#         super().__call__(x)
#         x = x[0]

#         return np.tanh(x)


# class SigmoidPrimitive(UnaryPrimitive):
#     def __init__(self, output_shape):
#         super().__init__(output_shape)
#         self.name = "sigmoid"
#         self.params = None
#         self.input_shapes = [output_shape]

#     def __call__(self, x: list) -> np.ndarray:
#         super().__call__(x)
#         x = x[0]

#         # Numerically stable sigmoid
#         out = np.empty_like(x)
#         pos = x >= 0
#         out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
#         exp_x = np.exp(x[~pos])
#         out[~pos] = exp_x / (1.0 + exp_x)
#         return out


# class ClipPrimitive(UnaryPrimitive):
#     def __init__(self, output_shape):
#         super().__init__(output_shape)
#         self.name = "clip"
#         self.params = {"lo": -1.0, "hi": 1.0}
#         self.input_shapes = [output_shape]

#     def __call__(self, x: list) -> np.ndarray:
#         super().__call__(x)
#         x = x[0]

#         lo = float(self.params["lo"])
#         hi = float(self.params["hi"])
#         if lo > hi:
#             lo, hi = hi, lo
#         return np.clip(x, lo, hi)


# class ShiftPrimitive(UnaryPrimitive):
#     def __init__(self, output_shape):
#         super().__init__(output_shape)
#         self.name = "shift"
#         self.params = {
#             "shift": np.random.randint(-output_shape[0] // 4, output_shape[0] // 4 + 1),
#             "pad_value": 0.0,
#         }
#         self.input_shapes = [[output_shape[0] - abs(self.params["shift"]), 0]]

#     def __call__(self, x: list) -> np.ndarray:
#         super().__call__(x)
#         x = x[0]

#         shift = int(self.params["shift"])
#         pad_value = float(self.params["pad_value"])
#         n = x.shape[0]

#         if n == 0 or shift == 0:
#             return x.copy()

#         out = np.full(x.shape, pad_value, dtype=np.float64)

#         if shift > 0:
#             out[shift:] = x[: n - shift]
#         else:
#             k = -shift
#             out[: n - k] = x[k:]

#         return out


# class ShrinkPrimitive(UnaryPrimitive):
#     def __init__(self, output_shape):
#         super().__init__(output_shape)
#         self.name = "shrink"
#         self.params = {"factor": np.random.uniform(0.75, 1.0)}
#         self.input_shapes = [
#             (output_shape[0] // self.params["factor"], output_shape[1])
#         ]

#     def __call__(self, x: list) -> np.ndarray:
#         super().__call__(x)
#         x = x[0]

#         x = np.asarray(x, dtype=np.float64)

#         # Ensure 2D
#         if x.ndim == 1:
#             x = x[:, None]

#         n, m = x.shape

#         factor = float(self.params["factor"])
#         out_len = max(1, int(n * factor))

#         xs_old, xs_new = _interp_grids(n, out_len)

#         output = np.empty((out_len, m), dtype=np.float64)

#         for j in range(m):
#             output[:, j] = np.interp(xs_new, xs_old, x[:, j])

#         return output


# class DiffPrimitive(UnaryPrimitive):
#     def __init__(self, output_shape):
#         super().__init__(output_shape)
#         self.name = "diff"
#         self.params = {"order": np.random.randint(1, 4)}
#         self.input_shapes = [(output_shape[0] + self.params["order"], output_shape[1])]

#     def __call__(self, x: list) -> np.ndarray:
#         super().__call__(x)
#         x = x[0]

#         return np.diff(x, n=self.params["order"], prepend=0.0)


# class CumsumPrimitive(UnaryPrimitive):
#     def __init__(self, output_shape):
#         super().__init__(output_shape)
#         self.name = "cumsum"
#         self.params = None
#         self.input_shapes = [output_shape]

#     def __call__(self, x: list) -> np.ndarray:
#         super().__call__(x)
#         x = x[0]

#         return np.cumsum(x, axis=0)


# class CummeanPrimitive(UnaryPrimitive):
#     def __init__(self, output_shape):
#         super().__init__(output_shape)
#         self.name = "cummean"
#         self.params = None
#         self.input_shapes = [output_shape]

#     def __call__(self, x: list) -> np.ndarray:
#         super().__call__(x)
#         x = x[0]

#         cumsum = np.cumsum(x, axis=0)
#         counts = np.arange(1, x.shape[0] + 1).reshape(-1, 1)
#         return cumsum / counts


# def _make_windows(x, window):
#     x = np.asarray(x, dtype=np.float64)
#     n = len(x)
#     w = max(1, int(window))

#     if n == 0:
#         return x, None, w

#     if w >= n:
#         return x, None, w

#     pad = w // 2
#     pad_width = tuple((pad, 0) if i == 0 else (0, 0) for i in range(x.ndim))
#     x_pad = np.pad(x, pad_width, mode="edge")
#     win = sliding_window_view(x_pad, w, axis=0)

#     return x, win, w


# class RollingStdPrimitive(UnaryPrimitive):
#     def __init__(self, output_shape):
#         super().__init__(output_shape)
#         self.name = "rolling_std"
#         self.params = {
#             "window": np.random.randint(2, output_shape[0] // 2 + 2),
#             "eps": 1e-12,
#         }
#         self.input_shapes = [output_shape]

#     def __call__(self, x: list) -> np.ndarray:
#         super().__call__(x)
#         x = x[0]

#         x, win, w = _make_windows(x, self.params["window"])
#         eps = self.params["eps"]

#         if len(x) == 0:
#             return x.copy()

#         if win is None:
#             return np.full_like(x, np.std(x) + eps, dtype=np.float64)

#         return np.std(win, axis=-1) + eps


# class RollingMedianPrimitive(UnaryPrimitive):
#     def __init__(self, output_shape):
#         super().__init__(output_shape)
#         self.name = "rolling_median"
#         self.params = {"window": np.random.randint(2, output_shape[0] // 2 + 2)}
#         self.input_shapes = [output_shape]

#     def __call__(self, x: list) -> np.ndarray:
#         super().__call__(x)
#         x = x[0]

#         x, win, w = _make_windows(x, self.params["window"])

#         if len(x) == 0:
#             return x.copy()

#         if win is None:
#             return np.full_like(x, np.median(x), dtype=np.float64)

#         return np.median(win, axis=-1)


# class RollingMinPrimitive(UnaryPrimitive):
#     def __init__(self, output_shape):
#         super().__init__(output_shape)
#         self.name = "rolling_min"
#         self.params = {"window": np.random.randint(2, output_shape[0] // 2 + 2)}
#         self.input_shapes = [output_shape]

#     def __call__(self, x: list) -> np.ndarray:
#         super().__call__(x)
#         x = x[0]

#         x, win, w = _make_windows(x, self.params["window"])

#         if len(x) == 0:
#             return x.copy()

#         if win is None:
#             return np.full_like(x, np.min(x), dtype=np.float64)

#         return np.min(win, axis=-1)


# class RollingMaxPrimitive(UnaryPrimitive):
#     def __init__(self, output_shape):
#         super().__init__(output_shape)
#         self.name = "rolling_max"
#         self.params = {"window": np.random.randint(2, output_shape[0] // 2 + 2)}
#         self.input_shapes = [output_shape]

#     def __call__(self, x: list) -> np.ndarray:
#         super().__call__(x)
#         x = x[0]

#         x, win, w = _make_windows(x, self.params["window"])

#         if len(x) == 0:
#             return x.copy()

#         if win is None:
#             return np.full_like(x, np.max(x), dtype=np.float64)

#         return np.max(win, axis=-1)


# class EMAPrimitive(UnaryPrimitive):
#     def __init__(self, output_shape):
#         super().__init__(output_shape)
#         self.name = "ema"
#         self.params = {"span": np.random.randint(2, output_shape[0] // 2 + 2)}
#         self.input_shapes = [output_shape]

#     def __call__(self, x: list) -> np.ndarray:
#         super().__call__(x)
#         x = x[0]

#         x = np.asarray(x, dtype=np.float64)
#         span = max(1, int(self.params["span"]))

#         if len(x) == 0:
#             return x.copy()

#         alpha = 2.0 / (span + 1.0)

#         out = np.empty_like(x)
#         out[0] = x[0]

#         for i in range(1, len(x)):
#             out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]

#         return out


# class GaussianSmoothPrimitive(UnaryPrimitive):
#     def __init__(self, output_shape):
#         super().__init__(output_shape)
#         self.name = "gaussian_smooth"
#         self.params = {
#             "window": np.random.randint(3, output_shape[0] // 2 + 2),
#             "sigma": None,
#         }
#         self.input_shapes = [output_shape]

#     def __call__(self, x: list) -> np.ndarray:
#         super().__call__(x)
#         x = x[0]

#         x = np.asarray(x, dtype=np.float64)

#         n, m = x.shape

#         window = max(1, int(self.params["window"]))

#         if n == 0:
#             return x.copy()

#         if window == 1:
#             return x.copy()

#         if window >= n:
#             mean = np.mean(x, axis=0, keepdims=True)
#             return np.repeat(mean, n, axis=0)

#         radius = window // 2

#         sigma = self.params["sigma"]
#         if sigma is None:
#             sigma = max(1e-6, window / 6.0)
#         sigma = max(float(sigma), 1e-6)

#         # Gaussian kernel
#         t = np.arange(-radius, radius + 1, dtype=np.float64)
#         kernel = np.exp(-(t * t) / (2.0 * sigma * sigma))
#         kernel /= np.sum(kernel) + 1e-12

#         # Apply per column
#         output = np.empty_like(x)

#         for j in range(m):
#             col = x[:, j]
#             col_pad = np.pad(col, (radius, radius), mode="edge")
#             output[:, j] = np.convolve(col_pad, kernel, mode="same")

#         return output.astype(np.float64)


# class RollingZScorePrimitive(UnaryPrimitive):
#     def __init__(self, output_shape):
#         super().__init__(output_shape)
#         self.name = "rolling_zscore"
#         self.params = {
#             "window": np.random.randint(2, output_shape[0] // 2 + 2),
#             "eps": 1e-12,
#         }
#         self.input_shapes = [output_shape]

#     def __call__(self, x: list) -> np.ndarray:
#         super().__call__(x)
#         x = x[0]

#         x = np.asarray(x, dtype=np.float64)
#         eps = self.params["eps"]
#         window = self.params["window"]

#         _, win, _ = _make_windows(x, window)

#         if len(x) == 0:
#             return x.copy()

#         if win is None:
#             mu = np.mean(x)
#             sd = np.std(x) + eps
#             return (x - mu) / sd

#         mu = np.mean(win, axis=-1)
#         sd = np.std(win, axis=-1) + eps

#         mu = mu[: len(x)]
#         sd = sd[: len(x)]

#         return (x - mu) / sd


# class DetrendPrimitive(UnaryPrimitive):
#     def __init__(self, output_shape):
#         super().__init__(output_shape)
#         self.name = "detrend"
#         self.params = {"window": np.random.randint(2, output_shape[0] // 2 + 2)}
#         self.input_shapes = [output_shape]

#     def __call__(self, x: list) -> np.ndarray:
#         super().__call__(x)
#         x = x[0]

#         x = np.asarray(x, dtype=np.float64)
#         window = self.params["window"]

#         _, win, _ = _make_windows(x, window)

#         if len(x) == 0:
#             return x.copy()

#         if win is None:
#             return x - np.mean(x)

#         trend = np.mean(win, axis=-1)
#         trend = trend[: len(x)]
#         return x - trend


# UNARY_PRIMITIVES = [
#     IdentityPrimitive,
#     ZscorePrimitive,
#     MinMaxPrimitive,
#     L2NormalizePrimitive,
#     AbsPrimitive,
#     SignPrimitive,
#     SquarePrimitive,
#     CubePrimitive,
#     SqrtPosPrimitive,
#     LogAbsPrimitive,
#     ExpClippedPrimitive,
#     ReluPrimitive,
#     TanhPrimitive,
#     SigmoidPrimitive,
#     ClipPrimitive,
#     ShiftPrimitive,
#     ShrinkPrimitive,
#     DiffPrimitive,
#     CumsumPrimitive,
#     CummeanPrimitive,
#     RollingStdPrimitive,
#     RollingMedianPrimitive,
#     RollingMinPrimitive,
#     RollingMaxPrimitive,
#     EMAPrimitive,
#     GaussianSmoothPrimitive,
#     RollingZScorePrimitive,
#     DetrendPrimitive,
# ]
