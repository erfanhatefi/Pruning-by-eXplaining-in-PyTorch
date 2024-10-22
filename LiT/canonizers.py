import torch.nn as nn

class SummationCanonizer:

    pass


class CanonizerReplaceModule:
    """
    Base class for canonizers that replace a module with another one.

    self.layer_map must contain as keys the module types to be replaced and as values a
    function that takes the original module as input and returns the new module.
    """

    def __init__(self, layer_map) -> None:
        
        self.layer_map = layer_map
        self.original_attributes = []

    def apply(self, parent: nn.Module, verbose=False) -> list:

        self._iterate_children(parent, verbose)

        # return a list to be conform with zennit
        return [self]

    def _iterate_children(self, parent: nn.Module, verbose):

        for name, child in parent.named_children():

            self._replace_attr(child, parent, name, verbose)
            self._iterate_children(child, verbose)

    def _replace_attr(self, module, parent, name, verbose):

        for source_type, init_fn in self.layer_map.items():
            if isinstance(module, source_type):

                # save original module to revert the composite in self.remove()
                self.original_attributes.append((parent, name, module))
                
                # create new module and attach it to parent as attribute
                new_module = init_fn(module)
                setattr(parent, name, new_module)

                if verbose:
                    print(f"-> Replaced module {name} {type(module)} with {type(new_module)}")

                return True

        return False

    def remove(self):
        #TODO: overwrite multihead forward pass instead of new module

        for parent, name, module in self.original_attributes:
            xai_module = getattr(parent, name)
            # delattr(parent, name)
            setattr(parent, name, module)
            del xai_module

        self.original_attributes = []



class CanonizerReplaceAttribute(CanonizerReplaceModule):
    """
    Base class for canonizers that replace a module's forward function.

    self.attr_map must contain as keys the module types and as values a
    dict with the attributes to be replaced as keys and the functions that create the new
    attributes as values. The functions take the original module as input and return the
    new attribute.

    Example: 
        attr_map = {
            torch.nn.MultiheadAttention: callable(module)
        }
    """

    def __init__(self, layer_map) -> None:
        
        self.layer_map = layer_map
        self.original_attributes = []

    def _replace_attr(self, module, _parent, name, verbose=False):

        for source_type, attr_map in self.layer_map.items():
            
            if isinstance(module, source_type):

                for attr_name, attr_fn in attr_map.items():

                    # save original module to revert the composite in self.remove()
                    if hasattr(module, attr_name):
                        self.original_attributes.append((module, attr_name, getattr(module, attr_name)))
                    else:
                        self.original_attributes.append((module, attr_name, None))
                    
                    setattr(module, attr_name, attr_fn(module))

                    if verbose:
                        print(f"-> Replaced attribute {attr_name} in module", name, type(module))

                return True

        return False

