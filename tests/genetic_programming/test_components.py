import sys
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt


def _find_repo_root(start: Path) -> Path:
    # Walk up until we find `src/genetic_programming/components.py`.
    for p in [start, *start.parents]:
        if (p / "src" / "genetic_programming" / "components.py").exists():
            return p
    raise RuntimeError(f"Could not locate repo root starting at {start}")


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
GP_DIR = REPO_ROOT / "src" / "genetic_programming"
if str(GP_DIR) not in sys.path:
    sys.path.insert(0, str(GP_DIR))

import components  # noqa: E402  # type: ignore[import-not-found]


class TestComponentsPrimitives(unittest.TestCase):
    def test_identity_ts_returns_copy_and_same_values(self) -> None:
        x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        y = components.identity_ts(x)
        self.assertTrue(np.array_equal(y, x))
        self.assertIsNot(y, x)

    def test_shift_ts_positive_and_negative(self) -> None:
        x = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        y_pos = components.shift_ts(x, shift=1, pad_value=-1.0)
        npt.assert_allclose(y_pos, np.array([-1.0, 10.0, 20.0], dtype=np.float64))

        y_neg = components.shift_ts(x, shift=-1, pad_value=-1.0)
        npt.assert_allclose(y_neg, np.array([20.0, 30.0, -1.0], dtype=np.float64))

    def test_resample_ts_output_length_and_endpoints(self) -> None:
        x = np.array([0.0, 3.0], dtype=np.float64)
        y = components.resample_ts(x, out_len=4)
        self.assertEqual(len(y), 4)
        npt.assert_allclose(y, np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64))

    def test_diff_ts_order_2(self) -> None:
        x = np.array([1.0, 3.0, 6.0], dtype=np.float64)
        y = components.diff_ts(x, order=2, pad_value=5.0)
        npt.assert_allclose(y, np.array([5.0, 5.0, 1.0], dtype=np.float64))

    def test_add_ts_aligns_different_lengths(self) -> None:
        x = np.array([0.0, 1.0, 2.0], dtype=np.float64)  # len=3
        y = np.array([0.0, 1.0], dtype=np.float64)  # len=2
        out = components.add_ts(x, y)
        # y is resampled to length 3: [0, 0.5, 1]
        npt.assert_allclose(out, np.array([0.0, 1.5, 3.0], dtype=np.float64))

    def test_concat_ts_length_and_values(self) -> None:
        x = np.array([1.0, 2.0], dtype=np.float64)
        y = np.array([3.0, 4.0, 5.0], dtype=np.float64)
        out = components.concat_ts(x, y)
        self.assertEqual(len(out), 5)
        npt.assert_allclose(out, np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64))

    def test_rolling_mean_ts_window_ge_len_returns_mean(self) -> None:
        x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        out = components.rolling_mean_ts(x, window=10)
        npt.assert_allclose(out, np.array([2.0, 2.0, 2.0], dtype=np.float64))


class TestComponentsRegistries(unittest.TestCase):
    def test_registries_contain_expected_keys(self) -> None:
        self.assertIn("identity", components.TS_UNARY_PRIMITIVES)
        self.assertIn("shift", components.TS_UNARY_PRIMITIVES)
        self.assertIn("add", components.TS_BINARY_PRIMITIVES)
        self.assertIn("mul", components.TS_BINARY_PRIMITIVES)
        self.assertIn("mean", components.TS_TO_SCALAR_PRIMITIVES)
        self.assertIn("std", components.TS_TO_SCALAR_PRIMITIVES)

        self.assertIn("unary_ts", components.ALL_PRIMITIVES)
        self.assertIn("binary_ts", components.ALL_PRIMITIVES)
        self.assertIn("ts_to_scalar", components.ALL_PRIMITIVES)


if __name__ == "__main__":
    unittest.main()

