"""
Time-series DSL primitives for program synthesis.

Conventions
-----------
* A "time series" is represented as a 1-D `numpy.ndarray` of floats.
* Most primitives return a 1-D time series (same length as the input).
* Some primitives can change length (frequency spectrum / segmentation); if you
  need strict same-length behavior, prefer the corresponding "*_up" variants.

The intent of this module is to provide a catalog of deterministic processing
functions that can be composed by a genetic program / expression tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

TimeSeries = np.ndarray
Scalar = float


def _as_1d_float(x: Any) -> TimeSeries:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def _safe_div(a: TimeSeries, b: TimeSeries, eps: float = 1e-12) -> TimeSeries:
    return a / (b + eps)


def _align_like(a: TimeSeries, b: TimeSeries) -> Tuple[TimeSeries, TimeSeries]:
    """
    Resample `b` to the length of `a` using linear interpolation.
    """

    a = _as_1d_float(a)
    b = _as_1d_float(b)

    if len(a) == len(b):
        return a, b

    out_len = len(a)
    if out_len == 0:
        return a, b
    if len(b) == 1:
        return a, np.full(out_len, float(b[0]), dtype=np.float64)

    xs_old = np.linspace(0.0, 1.0, num=len(b))
    xs_new = np.linspace(0.0, 1.0, num=out_len)
    b_res = np.interp(xs_new, xs_old, b)
    return a, b_res


def _pad_for_window(x: TimeSeries, window: int) -> Tuple[TimeSeries, int, int]:
    """
    Pad series so that a sliding window produces exactly `len(x)` outputs.
    """
    x = _as_1d_float(x)
    window = int(window)
    window = max(1, window)
    if len(x) == 0:
        return x, 0, 0

    left = window // 2
    right = window - 1 - left
    # Use edge padding to avoid introducing artificial zeros/outliers.
    x_pad = np.pad(x, (left, right), mode="edge")
    return x_pad, left, right


# ----------------------------
# Core (array-length-preserving)
# ----------------------------


def identity_ts(x: TimeSeries) -> TimeSeries:
    x = _as_1d_float(x)
    return x.copy()


def zscore_ts(x: TimeSeries, eps: float = 1e-12) -> TimeSeries:
    x = _as_1d_float(x)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    return (x - mu) / (sd + eps)


def minmax_ts(x: TimeSeries, eps: float = 1e-12) -> TimeSeries:
    x = _as_1d_float(x)
    mn = float(np.min(x))
    mx = float(np.max(x))
    return (x - mn) / ((mx - mn) + eps)


def l2_normalize_ts(x: TimeSeries, eps: float = 1e-12) -> TimeSeries:
    x = _as_1d_float(x)
    denom = float(np.linalg.norm(x) + eps)
    return x / denom


def abs_ts(x: TimeSeries) -> TimeSeries:
    x = _as_1d_float(x)
    return np.abs(x)


def sign_ts(x: TimeSeries) -> TimeSeries:
    x = _as_1d_float(x)
    return np.sign(x)


def square_ts(x: TimeSeries) -> TimeSeries:
    x = _as_1d_float(x)
    return x * x


def cube_ts(x: TimeSeries) -> TimeSeries:
    x = _as_1d_float(x)
    return x * x * x


def sqrt_pos_ts(x: TimeSeries) -> TimeSeries:
    x = _as_1d_float(x)
    return np.sqrt(np.maximum(x, 0.0))


def log_abs_ts(x: TimeSeries, eps: float = 1e-12) -> TimeSeries:
    x = _as_1d_float(x)
    return np.log(np.abs(x) + eps)


def exp_clipped_ts(x: TimeSeries, clip: float = 60.0) -> TimeSeries:
    x = _as_1d_float(x)
    return np.exp(np.clip(x, -clip, clip))


def relu_ts(x: TimeSeries) -> TimeSeries:
    x = _as_1d_float(x)
    return np.maximum(x, 0.0)


def tanh_ts(x: TimeSeries) -> TimeSeries:
    x = _as_1d_float(x)
    return np.tanh(x)


def sigmoid_ts(x: TimeSeries) -> TimeSeries:
    x = _as_1d_float(x)
    # Numerically stable sigmoid.
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def clip_ts(x: TimeSeries, lo: float = -1.0, hi: float = 1.0) -> TimeSeries:
    x = _as_1d_float(x)
    lo = float(lo)
    hi = float(hi)
    if lo > hi:
        lo, hi = hi, lo
    return np.clip(x, lo, hi)


def shift_ts(x: TimeSeries, shift: int, pad_value: float = 0.0) -> TimeSeries:
    """
    Shift by integer steps while keeping length.
    * shift > 0: delay (pad at start)
    * shift < 0: advance (pad at end)
    """
    x = _as_1d_float(x)
    n = len(x)
    shift = int(shift)
    if n == 0:
        return x.copy()
    if shift == 0:
        return x.copy()

    pad_value = float(pad_value)
    out = np.full(n, pad_value, dtype=np.float64)
    if shift > 0:
        out[shift:] = x[: n - shift]
    else:
        k = -shift
        out[: n - k] = x[k:]
    return out


def lag_ts(x: TimeSeries, lag: int) -> TimeSeries:
    """Alias for `shift_ts(x, lag, pad_value=0.0)`."""
    return shift_ts(x, lag, pad_value=0.0)


def resample_ts(x: TimeSeries, out_len: int) -> TimeSeries:
    """
    Resample to `out_len` with linear interpolation.
    """
    x = _as_1d_float(x)
    out_len = int(out_len)
    if out_len <= 0:
        return np.asarray([], dtype=np.float64)
    if len(x) == 0:
        return np.full(out_len, 0.0, dtype=np.float64)
    if len(x) == 1:
        return np.full(out_len, float(x[0]), dtype=np.float64)

    xs_old = np.linspace(0.0, 1.0, num=len(x))
    xs_new = np.linspace(0.0, 1.0, num=out_len)
    return np.interp(xs_new, xs_old, x)


def shrink_ts(x: TimeSeries, factor: float, out_len: Optional[int] = None) -> TimeSeries:
    """
    Shrink (compress) the series by `factor` (factor>1 -> fewer effective samples)
    and optionally resample back to `out_len` (defaults to original length).
    """
    x = _as_1d_float(x)
    factor = float(factor)
    if factor <= 0:
        factor = 1.0
    n = len(x)
    if out_len is None:
        out_len = n

    # First compress to a shorter sequence, then resample to out_len.
    compressed_len = max(1, int(round(n / factor)))
    x_comp = resample_ts(x, compressed_len)
    return resample_ts(x_comp, int(out_len))


def stretch_ts(x: TimeSeries, factor: float, out_len: Optional[int] = None) -> TimeSeries:
    """
    Stretch (expand) the series by `factor` (factor>1 -> longer effective samples)
    and optionally resample back to `out_len` (defaults to original length).
    """
    x = _as_1d_float(x)
    factor = float(factor)
    if factor <= 0:
        factor = 1.0
    n = len(x)
    if out_len is None:
        out_len = n

    expanded_len = max(1, int(round(n * factor)))
    x_exp = resample_ts(x, expanded_len)
    return resample_ts(x_exp, int(out_len))


def diff_ts(x: TimeSeries, order: int = 1, pad_value: float = 0.0) -> TimeSeries:
    """
    Discrete difference with same output length.
    """
    x = _as_1d_float(x)
    order = int(order)
    order = max(0, order)
    if order == 0:
        return x.copy()
    if len(x) == 0:
        return x.copy()
    if order >= len(x):
        return np.full_like(x, float(pad_value), dtype=np.float64)

    d = x
    for _ in range(order):
        d = np.diff(d)
    out = np.concatenate([np.full(order, float(pad_value), dtype=np.float64), d])
    return out


def cumsum_ts(x: TimeSeries) -> TimeSeries:
    x = _as_1d_float(x)
    return np.cumsum(x)


def cummean_ts(x: TimeSeries, eps: float = 1e-12) -> TimeSeries:
    x = _as_1d_float(x)
    n = len(x)
    if n == 0:
        return x.copy()
    idx = np.arange(1, n + 1, dtype=np.float64)
    return np.cumsum(x) / (idx + eps)


def rolling_mean_ts(x: TimeSeries, window: int) -> TimeSeries:
    x = _as_1d_float(x)
    if len(x) == 0:
        return x.copy()
    w = max(1, int(window))
    if w >= len(x):
        return np.full_like(x, float(np.mean(x)), dtype=np.float64)
    x_pad, _, _ = _pad_for_window(x, w)
    win = sliding_window_view(x_pad, w)
    return win.mean(axis=-1)


def rolling_std_ts(x: TimeSeries, window: int, eps: float = 1e-12) -> TimeSeries:
    x = _as_1d_float(x)
    if len(x) == 0:
        return x.copy()
    w = max(1, int(window))
    if w >= len(x):
        return np.full_like(x, float(np.std(x)), dtype=np.float64)
    x_pad, _, _ = _pad_for_window(x, w)
    win = sliding_window_view(x_pad, w)
    return win.std(axis=-1) + eps


def rolling_median_ts(x: TimeSeries, window: int) -> TimeSeries:
    x = _as_1d_float(x)
    if len(x) == 0:
        return x.copy()
    w = max(1, int(window))
    if w >= len(x):
        return np.full_like(x, float(np.median(x)), dtype=np.float64)
    x_pad, _, _ = _pad_for_window(x, w)
    win = sliding_window_view(x_pad, w)
    return np.median(win, axis=-1)


def rolling_min_ts(x: TimeSeries, window: int) -> TimeSeries:
    x = _as_1d_float(x)
    if len(x) == 0:
        return x.copy()
    w = max(1, int(window))
    if w >= len(x):
        return np.full_like(x, float(np.min(x)), dtype=np.float64)
    x_pad, _, _ = _pad_for_window(x, w)
    win = sliding_window_view(x_pad, w)
    return win.min(axis=-1)


def rolling_max_ts(x: TimeSeries, window: int) -> TimeSeries:
    x = _as_1d_float(x)
    if len(x) == 0:
        return x.copy()
    w = max(1, int(window))
    if w >= len(x):
        return np.full_like(x, float(np.max(x)), dtype=np.float64)
    x_pad, _, _ = _pad_for_window(x, w)
    win = sliding_window_view(x_pad, w)
    return win.max(axis=-1)


def ema_ts(x: TimeSeries, span: int) -> TimeSeries:
    """
    Exponential moving average.
    `span` maps to alpha via alpha=2/(span+1).
    """
    x = _as_1d_float(x)
    span = int(span)
    span = max(1, span)
    if len(x) == 0:
        return x.copy()
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def gaussian_smooth_ts(x: TimeSeries, window: int, sigma: Optional[float] = None) -> TimeSeries:
    """
    Simple Gaussian smoothing using a 1D kernel.
    """
    x = _as_1d_float(x)
    if len(x) == 0:
        return x.copy()
    window = int(window)
    window = max(1, window)
    if window == 1:
        return x.copy()
    if window >= len(x):
        return np.full_like(x, float(np.mean(x)), dtype=np.float64)
    radius = window // 2
    if sigma is None:
        sigma = max(1e-6, window / 6.0)
    sigma = float(sigma)
    if sigma <= 0:
        sigma = 1e-6

    t = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(t * t) / (2.0 * sigma * sigma))
    kernel = kernel / (np.sum(kernel) + 1e-12)

    x_pad = np.pad(x, (radius, radius), mode="edge")
    # Convolution with "valid" yields length == len(x)
    out = np.convolve(x_pad, kernel, mode="valid")
    return out.astype(np.float64, copy=False)


def rolling_zscore_ts(x: TimeSeries, window: int, eps: float = 1e-12) -> TimeSeries:
    x = _as_1d_float(x)
    mu = rolling_mean_ts(x, window)
    sd = np.maximum(rolling_std_ts(x, window, eps=eps), eps)
    return (x - mu) / sd


def detrend_ts(x: TimeSeries, window: int) -> TimeSeries:
    x = _as_1d_float(x)
    trend = rolling_mean_ts(x, window)
    return x - trend


# ----------------------------
# Pointwise/binary TS operations
# ----------------------------


def add_ts(x: TimeSeries, y: TimeSeries) -> TimeSeries:
    x, y = _align_like(x, y)
    return x + y


def sub_ts(x: TimeSeries, y: TimeSeries) -> TimeSeries:
    x, y = _align_like(x, y)
    return x - y


def mul_ts(x: TimeSeries, y: TimeSeries) -> TimeSeries:
    x, y = _align_like(x, y)
    return x * y


def div_ts(x: TimeSeries, y: TimeSeries, eps: float = 1e-12) -> TimeSeries:
    x, y = _align_like(x, y)
    return _safe_div(x, y, eps=eps)


def pow_ts(x: TimeSeries, y: TimeSeries, eps: float = 1e-12) -> TimeSeries:
    """
    Safe-ish power:
    * We shift by eps to avoid negative/zero issues for fractional powers.
    """
    x, y = _align_like(x, y)
    base = np.maximum(np.abs(x), eps)
    return np.power(base, y)


def max_ts(x: TimeSeries, y: TimeSeries) -> TimeSeries:
    x, y = _align_like(x, y)
    return np.maximum(x, y)


def min_ts(x: TimeSeries, y: TimeSeries) -> TimeSeries:
    x, y = _align_like(x, y)
    return np.minimum(x, y)


def concat_ts(x: TimeSeries, y: TimeSeries) -> TimeSeries:
    x = _as_1d_float(x)
    y = _as_1d_float(y)
    return np.concatenate([x, y])


# ----------------------------
# Frequency/segmentation transforms (may change length)
# ----------------------------


def fft_magnitude_ts(x: TimeSeries, take_half: bool = True, eps: float = 1e-12) -> TimeSeries:
    x = _as_1d_float(x)
    if len(x) == 0:
        return x.copy()
    spec = np.fft.fft(x)
    mag = np.abs(spec)
    if take_half:
        mag = mag[: len(mag) // 2 + 1]
    return np.log(mag + eps)


def fft_phase_ts(x: TimeSeries, take_half: bool = True) -> TimeSeries:
    x = _as_1d_float(x)
    if len(x) == 0:
        return x.copy()
    spec = np.fft.fft(x)
    phase = np.angle(spec)
    if take_half:
        phase = phase[: len(phase) // 2 + 1]
    return phase


def paa_ts(x: TimeSeries, segments: int) -> TimeSeries:
    """
    Piecewise Aggregate Approximation (PAA).
    Output length == `segments`.
    """
    x = _as_1d_float(x)
    segments = int(segments)
    segments = max(1, segments)
    n = len(x)
    if n == 0:
        return np.zeros(segments, dtype=np.float64)

    if n % segments == 0:
        return x.reshape((segments, -1)).mean(axis=1)

    # Unequal segment lengths (generic fallback)
    out = []
    for i in range(segments):
        start = int(i * n / segments)
        end = int((i + 1) * n / segments)
        if end <= start:
            out.append(float(x[start]))
        else:
            out.append(float(np.mean(x[start:end])))
    return np.asarray(out, dtype=np.float64)


def paa_up_ts(x: TimeSeries, segments: int, out_len: Optional[int] = None) -> TimeSeries:
    """
    PAA followed by linear interpolation back to `out_len` (default: len(x)).
    """
    x = _as_1d_float(x)
    if out_len is None:
        out_len = len(x)
    reduced = paa_ts(x, segments)
    return resample_ts(reduced, int(out_len))


def sax_ts(x: TimeSeries, segments: int = 4, alphabet_size: int = 4) -> TimeSeries:
    """
    Symbolic Aggregate Approximation (SAX).
    Returns integer symbols (as float array for numeric DSL compatibility).
    """
    x = _as_1d_float(x)
    segments = int(segments)
    segments = max(1, segments)
    alphabet_size = int(alphabet_size)
    alphabet_size = max(2, alphabet_size)
    if len(x) == 0:
        return np.zeros(segments, dtype=np.float64)

    # Normalize to z-score
    x_norm = (x - np.mean(x)) / (np.std(x) + 1e-12)
    paa_values = paa_ts(x_norm, segments)

    # Compute breakpoints for alphabet_size quantiles of standard normal
    try:
        from scipy.stats import norm  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "scipy is required for `sax_ts` breakpoints. Install scipy or avoid SAX."
        ) from e

    breakpoints = norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])
    symbols = np.digitize(paa_values, breakpoints)
    return symbols.astype(np.float64, copy=False)


def sax_up_ts(x: TimeSeries, segments: int = 4, alphabet_size: int = 4, out_len: Optional[int] = None) -> TimeSeries:
    x = _as_1d_float(x)
    if out_len is None:
        out_len = len(x)
    symbols = sax_ts(x, segments=segments, alphabet_size=alphabet_size)
    return resample_ts(symbols, int(out_len))


# ----------------------------
# Reductions (time series -> scalar)
# ----------------------------


def mean_ts(x: TimeSeries) -> Scalar:
    x = _as_1d_float(x)
    return float(np.mean(x)) if len(x) else 0.0


def std_ts(x: TimeSeries, eps: float = 1e-12) -> Scalar:
    x = _as_1d_float(x)
    return float(np.std(x) + eps) if len(x) else 0.0


def min_ts_scalar(x: TimeSeries) -> Scalar:
    x = _as_1d_float(x)
    return float(np.min(x)) if len(x) else 0.0


def max_ts_scalar(x: TimeSeries) -> Scalar:
    x = _as_1d_float(x)
    return float(np.max(x)) if len(x) else 0.0


def median_ts(x: TimeSeries) -> Scalar:
    x = _as_1d_float(x)
    return float(np.median(x)) if len(x) else 0.0


def energy_ts(x: TimeSeries) -> Scalar:
    x = _as_1d_float(x)
    return float(np.sum(x * x)) if len(x) else 0.0


def rms_ts(x: TimeSeries, eps: float = 1e-12) -> Scalar:
    x = _as_1d_float(x)
    if len(x) == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x) + eps))


def l1_norm_ts(x: TimeSeries) -> Scalar:
    x = _as_1d_float(x)
    return float(np.sum(np.abs(x))) if len(x) else 0.0


def l2_norm_ts(x: TimeSeries, eps: float = 1e-12) -> Scalar:
    x = _as_1d_float(x)
    return float(np.linalg.norm(x) + eps) if len(x) else 0.0


def peak_to_peak_ts(x: TimeSeries) -> Scalar:
    x = _as_1d_float(x)
    if len(x) == 0:
        return 0.0
    return float(np.max(x) - np.min(x))


def slope_ts(x: TimeSeries, eps: float = 1e-12) -> Scalar:
    """
    Slope of least-squares fit of y ~ a + b*t (t is index).
    """
    x = _as_1d_float(x)
    n = len(x)
    if n < 2:
        return 0.0
    t = np.arange(n, dtype=np.float64)
    t_mean = float(np.mean(t))
    x_mean = float(np.mean(x))
    denom = float(np.sum((t - t_mean) ** 2) + eps)
    b = float(np.sum((t - t_mean) * (x - x_mean)) / denom)
    return b


def skewness_ts(x: TimeSeries, eps: float = 1e-12) -> Scalar:
    x = _as_1d_float(x)
    if len(x) == 0:
        return 0.0
    mu = float(np.mean(x))
    sd = float(np.std(x) + eps)
    z = (x - mu) / sd
    return float(np.mean(z**3))


def kurtosis_ts(x: TimeSeries, eps: float = 1e-12) -> Scalar:
    x = _as_1d_float(x)
    if len(x) == 0:
        return 0.0
    mu = float(np.mean(x))
    sd = float(np.std(x) + eps)
    z = (x - mu) / sd
    return float(np.mean(z**4) - 3.0)  # excess kurtosis


def argmax_ts(x: TimeSeries) -> Scalar:
    x = _as_1d_float(x)
    return float(int(np.argmax(x))) if len(x) else 0.0


def argmin_ts(x: TimeSeries) -> Scalar:
    x = _as_1d_float(x)
    return float(int(np.argmin(x))) if len(x) else 0.0


# ----------------------------
# Registries
# ----------------------------


TS_UNARY_PRIMITIVES: Dict[str, Callable[..., Any]] = {
    # identity/normalization
    "identity": identity_ts,
    "zscore": zscore_ts,
    "minmax": minmax_ts,
    "l2norm": l2_normalize_ts,
    # pointwise ops
    "abs": abs_ts,
    "sign": sign_ts,
    "square": square_ts,
    "cube": cube_ts,
    "sqrt_pos": sqrt_pos_ts,
    "log_abs": log_abs_ts,
    "exp_clipped": exp_clipped_ts,
    "relu": relu_ts,
    "tanh": tanh_ts,
    "sigmoid": sigmoid_ts,
    "clip": clip_ts,
    # shifts/resampling
    "shift": shift_ts,
    "lag": lag_ts,
    "resample": resample_ts,
    "shrink": shrink_ts,
    "stretch": stretch_ts,
    # differences/integrals
    "diff": diff_ts,
    "cumsum": cumsum_ts,
    "cummean": cummean_ts,
    # rolling/window ops
    "rolling_mean": rolling_mean_ts,
    "rolling_std": rolling_std_ts,
    "rolling_median": rolling_median_ts,
    "rolling_min": rolling_min_ts,
    "rolling_max": rolling_max_ts,
    "ema": ema_ts,
    "gaussian_smooth": gaussian_smooth_ts,
    "rolling_zscore": rolling_zscore_ts,
    # detrending
    "detrend": detrend_ts,
    # frequency/symbolic transforms (may change length)
    "fft_mag": fft_magnitude_ts,
    "fft_phase": fft_phase_ts,
    "paa": paa_ts,
    "paa_up": paa_up_ts,
    "sax": sax_ts,
    "sax_up": sax_up_ts,
}


TS_BINARY_PRIMITIVES: Dict[str, Callable[..., Any]] = {
    "add": add_ts,
    "sub": sub_ts,
    "mul": mul_ts,
    "div": div_ts,
    "pow": pow_ts,
    "max": max_ts,
    "min": min_ts,
    "concat": concat_ts,
}


TS_TO_SCALAR_PRIMITIVES: Dict[str, Callable[..., Any]] = {
    "mean": mean_ts,
    "std": std_ts,
    "min": min_ts_scalar,
    "max": max_ts_scalar,
    "median": median_ts,
    "energy": energy_ts,
    "rms": rms_ts,
    "l1_norm": l1_norm_ts,
    "l2_norm": l2_norm_ts,
    "peak_to_peak": peak_to_peak_ts,
    "slope": slope_ts,
    "skewness": skewness_ts,
    "kurtosis": kurtosis_ts,
    "argmax": argmax_ts,
    "argmin": argmin_ts,
}


ALL_PRIMITIVES: Dict[str, Dict[str, Callable[..., Any]]] = {
    "unary_ts": TS_UNARY_PRIMITIVES,
    "binary_ts": TS_BINARY_PRIMITIVES,
    "ts_to_scalar": TS_TO_SCALAR_PRIMITIVES,
}


__all__ = [
    "TimeSeries",
    "Scalar",
    "TS_UNARY_PRIMITIVES",
    "TS_BINARY_PRIMITIVES",
    "TS_TO_SCALAR_PRIMITIVES",
    "ALL_PRIMITIVES",
]
