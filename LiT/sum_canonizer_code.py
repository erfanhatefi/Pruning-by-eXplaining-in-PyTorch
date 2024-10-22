import ast_comments as ast
import inspect
import textwrap

class SumReplacer(ast.NodeTransformer):

    def __init__(self, ignore_classes=[]) -> None:
        super().__init__()
        self.ignore_classes = ignore_classes
        self.contains_sum = False
        self.counter = 0
        
    def create_comment(self, string_code):
        comment = ast.Comment(f"### {string_code} <--- EDITED BY LiT", inline=True)
        return comment
    
    def visit_ClassDef(self, node):

        for ignore in self.ignore_classes:
            if node.name == ignore.__class__.__name__:
                return node

        self.contains_sum = False
        self.generic_visit(node)
        if self.contains_sum:
            for body_node in node.body:
                if isinstance(body_node, ast.FunctionDef) and body_node.name == '__init__':
                    
                    new_node = ast.parse("self.sum_layer_lit = SumLiT()")
                    comment = self.create_comment("")
                    new_node = ast.Module(body=[new_node, comment], type_ignores=[])
                    body_node.body.append(new_node)
                    self.counter += 1
        
        self.contains_sum = False
        return node

    def visit_AugAssign(self, node):

        if isinstance(node.op, ast.Add):
            if isinstance(node.value, ast.Tuple) or isinstance(node.value, ast.List):
                return node
            
            self.contains_sum = True

            value = ast.Call(
                func=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='sum_layer_lit', ctx=ast.Load()),
                args=[ast.Name(id=node.target.id, ctx=ast.Load()), node.value],
                keywords=[]
            )
            new_node = ast.Assign(targets=[ast.Name(id=node.target.id, ctx=ast.Store())], value=value)
            comment = self.create_comment(ast.unparse(node))
            new_node = ast.Module(body=[new_node, comment], type_ignores=[])

            self.counter += 1

            return new_node
        return node
    
    def visit_BinOp(self, node):

        if isinstance(node.op, ast.Add):
            if isinstance(node.left, ast.Tuple) or isinstance(node.right, ast.Tuple) or isinstance(node.left, ast.List) or isinstance(node.right, ast.List):
                return node
            
            self.contains_sum = True

            new_node = ast.Call(
                func=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='sum_layer_lit', ctx=ast.Load()),
                args=[node.left, node.right],
                keywords=[]
            )
            comment = self.create_comment(ast.unparse(node))
            new_node = ast.Module(body=[new_node, comment], type_ignores=[])

            self.counter += 1
            
            return new_node
        return node


def sum_canonizer_source_code(module_object, ignore_classes=[], file_name=None):

    replacer = SumReplacer(ignore_classes)

    source_code = inspect.getsource(module_object) 
    source_code = f"from LiT.layers import SumLiT ### <--- EDITED BY LiT\n" + source_code
    # remove indentation if available so that ast.parse is happy
    source_code = textwrap.dedent(source_code)
    # parse source code into AST
    tree = ast.parse(source_code)
    # replace addition nodes with sum_lit_layer
    tree = replacer.visit(tree)
    # fix missing locations in tree so that AST is happy
    ast.fix_missing_locations(tree)

    # save
    new_source_code = ast.unparse(tree)
    if file_name is None:
        file_name = module_object.__name__.split('.')[-1] + "_LiT_" + '.py'

    with open(file_name, 'w') as f:
        f.write(new_source_code)

    print("Number of modifications:", replacer.counter + 1)




if __name__ == "__main__":

    from torchvision.models import resnet 

    sum_canonizer_source_code(resnet)