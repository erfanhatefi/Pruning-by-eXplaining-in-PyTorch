import ast
import torchvision.models.resnet as resnet
import inspect
import textwrap
import astor
import astunparse
import sys
from pxp.LiT.layers import SumLiT
from torch import Tensor
import os


class SumReplacer(ast.NodeTransformer):

    def visit_BinOp(self, node):

        if isinstance(node.op, ast.Add):
            if (
                isinstance(node.left, ast.Tuple)
                or isinstance(node.right, ast.Tuple)
                or isinstance(node.left, ast.List)
                or isinstance(node.right, ast.List)
            ):
                return node

            new_node = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr="sum_lit_layer",
                    ctx=ast.Load(),
                ),
                args=[node.left, node.right],
                keywords=[],
            )
            return new_node
        return node

    def visit_AugAssign(self, node):

        if isinstance(node.op, ast.Add):
            if isinstance(node.value, ast.Tuple) or isinstance(node.value, ast.List):
                return node

            value = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr="sum_lit_layer",
                    ctx=ast.Load(),
                ),
                args=[ast.Name(id=node.target.id, ctx=ast.Load()), node.value],
                keywords=[],
            )
            new_node = ast.Assign(
                targets=[ast.Name(id=node.target.id, ctx=ast.Store())], value=value
            )
            return new_node
        return node

    # def generic_visit(self, node: AST) -> AST:

    #     if isinstance(node, ast.ClassDef):
    #         print("ClassDef", node)
    #     return super().generic_visit(node)


def sum_canonizer(model, namespace, ignore_layers=[], save=False):

    for _, module in model.named_modules():
        for ignore in ignore_layers:
            if isinstance(module, ignore):
                continue

        if hasattr(module, "sum_lit_layer"):
            raise AttributeError(
                "Please rename your module attribute 'sum_lit_layer' to something else."
            )
        else:
            # add sum_lit_layer to module
            setattr(module, "sum_lit_layer", SumLiT())

        # Get the source code of the instance's forward method
        source_code = inspect.getsource(module.forward)
        # remove indentation if available so that ast.parse is happy
        source_code = textwrap.dedent(source_code)
        # parse source code into AST
        tree = ast.parse(source_code)
        # replace addition nodes with sum_lit_layer
        tree = replacer.visit(tree)
        # fix missing locations in tree so that AST is happy
        ast.fix_missing_locations(tree)

        if save:
            code_string = astunparse.unparse(tree)
            # dump into ending of file
            with open("sum_canonized.py", "a") as f:
                f.write(f"\n\n{module.__class__.__name__}\n")
                f.write(code_string)

        # compile tree into Python Code
        compiled = compile(tree, filename="<ast>", mode="exec")
        # now, execute (load) the code into the namespace of my_instance
        # exec(compiled, model.__dict__)
        exec(compiled, namespace.__dict__)
        # finally, rebind the "self" argument of the method to my_instance
        module.forward = module.forward.__get__(module)


if __name__ == "__main__":

    model = resnet.resnet18()

    replacer = SumReplacer()

    for name, module in model.named_modules():
        if isinstance(module, resnet.BasicBlock):
            if hasattr(module, "sum_lit_layer"):
                raise AttributeError(
                    "Please rename your module attribute 'sum_lit_layer' to something else."
                )

            # add sum_lit_layer to module
            setattr(module, "sum_lit_layer", SumLiT())
            # Get the source code of the instance's forward method
            source_code = inspect.getsource(module.forward)
            # remove indentation if available so that ast.parse is happy
            source_code = textwrap.dedent(source_code)
            # parse source code into AST
            tree = ast.parse(source_code)
            # replace addition nodes with sum_lit_layer
            tree = replacer.visit(tree)
            # fix missing locations in tree so that AST is happy
            ast.fix_missing_locations(tree)

            code_string = astunparse.unparse(tree)

            # dump into ending of file
            with open("test.py", "a") as f:
                f.write(f"\n\n{module.__class__.__name__}\n")
                f.write(code_string)

            # compile tree into Python Code
            compiled = compile(tree, filename="<ast>", mode="exec")
            # now, execute (load) the code into the namespace of my_instance
            # exec(compiled, model.__dict__)
            exec(compiled)
            # finally, rebind the "self" argument of the method to my_instance
            module.forward = module.forward.__get__(module)
