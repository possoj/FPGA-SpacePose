"""
Copyright (c) 2024 Julien Posso
"""

from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.basic import get_by_name


class AbsorbConsecutiveTransposes(Transformation):
    """
    Remove (Transpose -> Transpose) patterns when the input and output
    of the pattern have the same layout.
    Current node = Transpose. Can be a fork, but cannot be a join. Join nodes are only Add/Sub or Concat operations.
    If Next node is Transpose, it can ba a fork, but cannot be a join
    """

    @staticmethod
    def are_opposite_permutations(perms1, perms2):
        if len(perms1) != len(perms2):
            return False
        assert 0 <= max(perms2) < len(perms2), "invalid permutation"
        assert 0 <= max(perms1) < len(perms1), "invalid permutation"

        for i, p in enumerate(perms2):
            if perms1[p] != i:
                return False

        return True

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        for current_node in graph.node:

            # Check that the current node is a Transpose node
            if current_node.op_type != "Transpose":
                continue

            # Find current node successors and check that they are Transpose nodes
            next_nodes = model.find_direct_successors(current_node)
            if not all(next_node.op_type == "Transpose" for next_node in next_nodes):
                continue

            # Check that current node and successors are opposite Transpose nodes (NHWC and NCHW)
            perms1 = list(get_by_name(current_node.attribute, "perm").ints)
            for next_node in next_nodes:
                perms2 = list(get_by_name(next_node.attribute, "perm").ints)
                if not self.are_opposite_permutations(perms1, perms2):
                    raise Exception(f"Error, seems both current and next node are transpose but implement the same "
                                    f"transformation:\npermute 1 = {perms1}, permute 2 = {perms2}")

            # Find all the successors of the next nodes (potentially several successors if the next node is a fork)
            # and bind the input of the successors to the input of the current node
            # Transpose nodes have only one input and one output (even if they are fork nodes)
            for next_node in next_nodes:
                successors = model.find_direct_successors(next_node)
                for successor in successors:
                    for idx, successor_input in enumerate(successor.input):
                        if successor_input == next_node.output[0]:
                            successor.input[idx] = current_node.input[0]

            # remove transposes
            graph.node.remove(current_node)
            for next_node in next_nodes:
                graph.node.remove(next_node)

            graph_modified = True

        if graph_modified:
            model = model.transform(InferDataTypes())
            model = model.transform(InferShapes())
        return model, graph_modified
