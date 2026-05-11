from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional, Tuple
from sympy import Symbol, solve

import numpy as np

from primitives import (
    INPUT_COLUMNS,
    INPUT_ROWS,
    UNARY_PRIMITIVES,
    IdentityPrimitive,
    InputPrimitive,
    Primitive,
    Shape,
)

AstNodeType = Literal["input", "unary", "binary"]


@dataclass()
class TsAstProgram:
    root: Primitive
    target_length: int
    constraint_solution: dict = dict


def _make_unary_node():
    primitive_const = np.random.choice(UNARY_PRIMITIVES)
    return primitive_const()


def create_ast(
    max_depth: int = 4,
    target_length: int = 256,
    unary_ops: Optional[list[str]] = None,
    binary_ops: Optional[list[str]] = None,
) -> TsAstProgram:
    """
    Create a random AST program.

    The AST is guaranteed to output a time series of length `target_length`
    after evaluation.
    """
    rng = np.random.default_rng()

    if rng is None:
        rng = np.random.default_rng()

    target_length = int(target_length)
    if target_length <= 0:
        raise ValueError("`target_length` must be a positive integer.")
    max_depth = int(max_depth)
    if max_depth < 0:
        raise ValueError("`max_depth` must be >= 0.")

    p_unary = 1.0

    def build(d: int) -> Primitive:

        node: Primitive = _make_unary_node()

        if d <= 0:
            return InputPrimitive()

        child_node = build(d - 1)
        node.add_child(child_node)

        return node

    def collect_constraints(constraints: list, node: Primitive):
        node_constraints = node.infer_parameters()
        constraints.extend(node_constraints)

        for child in node.children:
            collect_constraints(constraints, child)

    root = build(max_depth)
    root.add_output_constraint(1, target_length)

    constraints = []
    collect_constraints(constraints, root)

    solution = solve(constraints, dict=True)
    solution = solution[0]

    program = TsAstProgram(root=root, target_length=target_length)
    program.constraint_solution = solution

    return program


def evaluate_ast(program: TsAstProgram, x: np.ndarray) -> np.ndarray:
    """
    Evaluate `program` on input series `x`.
    """

    def eval_node(node: Primitive, x: np.ndarray) -> np.ndarray:
        if node.children == []:
            return node([x])
        else:
            runtime_parameters = {
                INPUT_ROWS: x.shape[1],
                INPUT_COLUMNS: x.shape[0],
            }

            solution = {
                **program.constraint_solution,
                **runtime_parameters,
            }
            node.constraint_solution = solution
            child_values = []
            for child in node.children:
                child_value = eval_node(child, x)
                child_values.append(child_value)
            return node(child_values)

    return eval_node(program.root, x)


if __name__ == "__main__":
    for _ in range(20):
        program = create_ast(max_depth=3, target_length=256)
        x = np.random.rand(1200)
        x = x.reshape(-1, 1)
        output = evaluate_ast(program, x)
        print(output.shape)
