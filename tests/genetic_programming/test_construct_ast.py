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
import construct_ast  # noqa: E402  # type: ignore[import-not-found]


class TestConstructAst(unittest.TestCase):
    def test_create_ast_invalid_target_length(self) -> None:
        with self.assertRaises(ValueError):
            construct_ast.create_ast(target_length=0)

    def test_create_ast_unary_ops_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            construct_ast.create_ast(target_length=8, max_depth=1, unary_ops=[])

    def test_create_ast_max_depth_zero_builds_resample_only(self) -> None:
        program = construct_ast.create_ast(max_depth=0, target_length=4, rng=np.random.default_rng(0))
        self.assertEqual(program.target_length, 4)
        self.assertEqual(program.root.node_type, "unary")
        self.assertEqual(program.root.op_name, "resample")
        self.assertIsNotNone(program.root.params)
        self.assertEqual(program.root.params.get("out_len"), 4)

        x = np.array([0.0, 3.0], dtype=np.float64)
        out = construct_ast.evaluate_ast(program, x)
        expected = components.resample_ts(x, out_len=4)
        npt.assert_allclose(out, expected)

    def test_evaluate_ast_output_length_for_variable_length_primitive(self) -> None:
        root = construct_ast.AstNode(
            node_type="unary",
            op_name="fft_mag",
            params={"take_half": True},
            child=construct_ast.AstNode(node_type="input"),
        )
        program = construct_ast.TsAstProgram(root=root, target_length=20)

        x = np.linspace(0.0, 1.0, 10, dtype=np.float64)
        out = construct_ast.evaluate_ast(program, x)
        self.assertEqual(len(out), 20)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_evaluate_ast_reshapes_2d_input(self) -> None:
        program = construct_ast.TsAstProgram(
            root=construct_ast.AstNode(node_type="input"),
            target_length=4,
        )
        x2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        out = construct_ast.evaluate_ast(program, x2d)

        flattened = x2d.reshape(-1)
        expected = components.resample_ts(flattened, out_len=4)
        npt.assert_allclose(out, expected)

    def test_create_ast_deterministic_with_same_seed(self) -> None:
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        p1 = construct_ast.create_ast(max_depth=3, target_length=16, p_unary=0.6, rng=rng1)
        p2 = construct_ast.create_ast(max_depth=3, target_length=16, p_unary=0.6, rng=rng2)
        self.assertEqual(p1.target_length, p2.target_length)
        self.assertEqual(p1.root, p2.root)

    def test_evaluate_ast_binary_add(self) -> None:
        x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        left = construct_ast.AstNode(node_type="unary", op_name="identity", params=None, child=construct_ast.AstNode(node_type="input"))
        right = construct_ast.AstNode(node_type="unary", op_name="identity", params=None, child=construct_ast.AstNode(node_type="input"))
        root = construct_ast.AstNode(node_type="binary", op_name="add", params=None, left=left, right=right)
        program = construct_ast.TsAstProgram(root=root, target_length=len(x))

        out = construct_ast.evaluate_ast(program, x)
        npt.assert_allclose(out, 2.0 * x)

    def test_compile_ast_matches_evaluate_ast_simple(self) -> None:
        program = construct_ast.TsAstProgram(
            root=construct_ast.AstNode(
                node_type="unary",
                op_name="resample",
                params={"out_len": 4},
                child=construct_ast.AstNode(node_type="input"),
            ),
            target_length=4,
        )
        compiled = construct_ast.compile_ast(program)

        x2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        out_compiled = compiled(x2d)
        out_eval = construct_ast.evaluate_ast(program, x2d)
        npt.assert_allclose(out_compiled, out_eval)

    def test_compile_ast_matches_evaluate_ast_generated(self) -> None:
        rng = np.random.default_rng(0)
        program = construct_ast.create_ast(max_depth=3, target_length=16, p_unary=0.6, rng=rng)
        compiled = construct_ast.compile_ast(program)

        x = np.linspace(0.0, 1.0, 32, dtype=np.float64)
        out_compiled = compiled(x)
        out_eval = construct_ast.evaluate_ast(program, x)
        npt.assert_allclose(out_compiled, out_eval)


if __name__ == "__main__":
    unittest.main()

