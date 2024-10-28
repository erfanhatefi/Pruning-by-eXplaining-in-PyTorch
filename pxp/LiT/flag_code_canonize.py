import re
import inspect
import timm.models.vision_transformer as vit

def add_comment_to_code(code, operations, irgnore_moduls=[]):
    lines = code.split('\n')

    in_class_to_analyze = False
    updated_lines = []
    counter = 0

    for line in lines:
        if any(f"class {m.__class__.__name__}" in line for m in irgnore_moduls):
            in_class_to_analyze = False
        elif 'class' in line:
            in_class_to_analyze = True

        if in_class_to_analyze:
            for op in operations:
                if op in line:
                    line = line.rstrip() + '  ### FLAG\n'
                    counter += 1
                    break
        
        updated_lines.append(line)

    with open("test.py", 'w') as file:
        file.write('\n'.join(updated_lines))

    print("Lines that potentially need to be canonized", counter)

source_code = inspect.getsource(vit)
add_comment_to_code(source_code, ["torch.bmm", "torch.matmul", "+", "+=", "*", "softmax"])