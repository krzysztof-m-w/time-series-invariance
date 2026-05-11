from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional, Tuple

import numpy as np

from primitives import UNARY_PRIMITIVES, Primitive

AstNodeType = Literal["input", "unary", "binary"]


@dataclass(frozen=True)
class AstNode:
    node_type: AstNodeType
    # For unary: op_name + child
    primitive: Optional[Primitive] = None
    children: list["AstNode"] = None


@dataclass(frozen=True)
class TsAstProgram:
    root: AstNode
    target_length: int

def _make_input_node() -> AstNode:
    return AstNode(node_type="input")

def _make_unary_node(output_shape: Tuple[int, int]) -> AstNode:
    return AstNode(
        node_type="unary",
        primitive=UNARY_PRIMITIVES[np.random.randint(0, len(UNARY_PRIMITIVES))](output_shape),
        children=[]
    )


def _make_binary_node(op_name: str, params: Dict[str, Any], left: AstNode, right: AstNode) -> AstNode:
    return AstNode(
        node_type="binary",
        op_name=op_name,
        params=params,
        left=left,
        right=right,
    )


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


    # Base: always resample the input to fixed length.
    input_node = _make_input_node()
    base = _make_unary_node(output_shape=(target_length,1))

    p_unary = 1.0

    def build(d: int, output_shape: Tuple[int, int]) -> AstNode:

        node = _make_unary_node(output_shape)

        if d <= 0:
            return input_node
        
        for input_shape in node.primitive.input_shapes:
            child_node = build(d - 1, input_shape)
            node.children.append(child_node)

        return node

    root = build(max_depth, (target_length,1))
    return TsAstProgram(root=root, target_length=target_length)


def evaluate_ast(program: TsAstProgram, x: np.ndarray) -> np.ndarray:
    """
    Evaluate `program` on input series `x`.
    """

    def eval_node(node: AstNode, x: np.ndarray) -> np.ndarray:
        if node.node_type == "input":
            return x
        elif node.node_type == "unary":
            child_values = []
            for child in node.children:
                child_value = eval_node(child, x)
                if child.node_type != "input" and child_value.shape != child.primitive.output_shape:
                    raise ValueError(f"Child node output shape {child_value.shape} does not match expected shape {child.primitive.output_shape}.")
                child_values.append(child_value)
            return node.primitive(child_values)
        
    return eval_node(program.root, x)
        

if __name__ == "__main__":
    for _ in range(20):
        program = create_ast(max_depth=3, target_length=256)
        x = np.random.rand(256)
        x = x.reshape(-1, 1)
        output = evaluate_ast(program, x)
        print(output.shape)
