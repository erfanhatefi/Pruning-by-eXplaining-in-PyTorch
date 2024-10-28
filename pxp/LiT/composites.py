import torch.nn as nn
import pxp.LiT.rules as rules
import weakref
import gc


class Composite:
    """
    Base class for composites. A composite is a collection of rules that are applied to a model.

    self.layer_map is a dict of the form {layer_type: rule} where layer_type is a torch.nn.Module and
    rule is a LiT rule.
    """

    def __init__(self, layer_map, canonizers=[], zennit_composite=None) -> None:

        self.layer_map = layer_map
        self.original_modules = []

        self.canonizers = canonizers
        self.canonizer_instances = []

        for c in canonizers:
            if isinstance(c, type):
                raise ValueError(
                    f"You must call the canonizer {c}(). You passed the class instead of an instance."
                )

        self.zennit_composite = zennit_composite

    def register(self, parent: nn.Module, verbose=False) -> None:

        for canonizer in self.canonizers:

            try:
                # LiT canonizers
                instances = canonizer.apply(parent, verbose)
            except TypeError:
                # zennit canonizers dont have verbose argument
                instances = canonizer.apply(parent)

            self.canonizer_instances.extend(instances)

        self._iterate_children(parent, self.layer_map, verbose)

        # register an optional zennit composite
        # if self.zennit_composite is not None:
        #     if verbose:
        #         print("-> register ZENNIT composite", self.zennit_composite)
        #     self.zennit_composite.register(parent)

        # register an optional zennit composite
        if self.zennit_composite:
            if verbose:
                print("-> register ZENNIT composite", self.zennit_composite)
            self.zennit_composite.register(parent)

    def _iterate_children(self, parent: nn.Module, rule_dict, verbose):

        for name, child in parent.named_children():

            self._attach_rule(child, parent, name, rule_dict, verbose)
            self._iterate_children(child, rule_dict, verbose)

    def _attach_rule(self, child, parent, name, rule_dict, verbose):

        for layer_type, rule in rule_dict.items():

            if (isinstance(layer_type, str) and name.endswith(layer_type)) or (
                not isinstance(layer_type, str) and isinstance(child, layer_type)
            ):

                # replace module with xai_module and attach it to parent as attribute
                xai_module = rule(child)
                setattr(parent, name, xai_module)

                # save original module to revert the composite in self.remove()
                self.original_modules.append((parent, name, child))

                if verbose:
                    print(
                        f"-> Explained module {name} {type(child)} with {type(xai_module)}"
                    )

                return True

        if verbose:
            print(f"?? Module {name} {type(child)} has no rule")

        return False

    def remove(self):

        for parent, name, module in self.original_modules:
            rule = getattr(parent, name)
            # delattr(parent, name)
            setattr(parent, name, module)
            del rule

        for instance in self.canonizer_instances:
            instance.remove()

        self.original_modules = []
        self.canonizer_instances = []

        if self.zennit_composite is not None:
            self.zennit_composite.remove()

    def context(self, module, verbose=False):

        return CompositeContext(module, self, verbose)


class CompositeNEW:
    """
    Base class for composites. A composite is a collection of rules that are applied to a model.

    self.layer_map is a dict of the form {layer_type: rule} where layer_type is a torch.nn.Module and
    rule is a LiT rule.
    """

    def __init__(self, layer_map, canonizers=[], zennit_composite=None) -> None:

        self.layer_map = layer_map
        self.original_modules = []

        self.canonizers = canonizers
        self.canonizer_instances = []

        for c in canonizers:
            if isinstance(c, type):
                raise ValueError(
                    f"You must call the canonizer {c}(). You passed the class instead of an instance."
                )

        self.zennit_composite = zennit_composite

    def register(self, parent: nn.Module, verbose=False) -> None:

        if len(self.original_modules) > 0:
            raise RuntimeError("Composite already registered. Please remove it first.")

        for canonizer in self.canonizers:

            try:
                # LiT canonizers
                instances = canonizer.apply(parent, verbose)
            except TypeError:
                # zennit canonizers dont have verbose argument
                instances = canonizer.apply(parent)

            self.canonizer_instances.extend(instances)

        self._iterate_children(parent, self.layer_map, verbose)

        # register an optional zennit composite
        if self.zennit_composite is not None:
            if verbose:
                print("-> register ZENNIT composite", self.zennit_composite)
            self.zennit_composite.register(parent)

    def _iterate_children(self, parent: nn.Module, rule_dict, verbose):

        for name, child in parent.named_children():

            self._attach_rule(child, parent, name, rule_dict, verbose)
            self._iterate_children(child, rule_dict, verbose)

    def _attach_rule(self, child, parent, name, rule_dict, verbose):

        for layer_type, rule in rule_dict.items():

            if layer_type == name or (
                not isinstance(layer_type, str) and isinstance(child, layer_type)
            ):

                if isinstance(rule, dict):
                    # rule is dict with sub-rules
                    self._iterate_children(child, rule, verbose)

                else:
                    # replace module with xai_module and attach it to parent as attribute
                    xai_module = rule(child)
                    setattr(parent, name, xai_module)

                    # save original module to revert the composite in self.remove()
                    self.original_modules.append((parent, name, child))

                    if verbose:
                        print(
                            f"-> Explained module {name} {type(child)} with {type(xai_module)}"
                        )

                    return True

        if verbose:
            print(f"?? Moduke {name} {type(child)} has no rule")

        return False

    def remove(self):

        for parent, name, module in self.original_modules:
            rule = getattr(parent, name)
            setattr(parent, name, module)
            del rule

        for instance in self.canonizer_instances:
            instance.remove()

        self.canonizer_instances = []
        self.original_modules = []

        if self.zennit_composite is not None:
            self.zennit_composite.remove()

    def context(self, module, verbose=False):

        return CompositeContext(module, self, verbose)


class CompositeContext:
    """A context object to register a composite in a context and remove the associated hooks and canonizers afterwards.

    Parameters
    ----------
    module: :py:class:`torch.nn.Module`
        The module to which `composite` should be registered.
    composite: :py:class:`zennit.core.Composite`
        The composite which shall be registered to `module`.
    """

    def __init__(self, module, composite, verbose):
        self.module = module
        self.composite = composite
        self.verbose = verbose

    def __enter__(self):
        self.composite.register(self.module, self.verbose)
        return self.module

    def __exit__(self, exc_type, exc_value, traceback):
        self.composite.remove()
        return False


# ####################################################
# ####################################################
# ################# EXTRA COMPOSITES #################
# ####################################################
# ####################################################


# class NameMapComposite(Composite):
#     """A Composite for which hooks are specified by a mapping from module names to hooks.

#     Parameters
#     ----------
#     name_map: `list[tuple[tuple[str, ...], Hook]]`
#         A mapping as a list of tuples, with a tuple of applicable module names and a Hook.
#     canonizers: list[:py:class:`zennit.canonizers.Canonizer`], optional
#         List of canonizer instances to be applied before applying hooks.
#     """

#     def __init__(self, name_map, canonizers=[], zennit_composite=None) -> None:
#         self.name_map = name_map
#         self.original_modules = []

#         self.canonizers = canonizers
#         self.canonizer_instances = []

#         for c in canonizers:
#             if isinstance(c, type):
#                 raise ValueError(
#                     f"You must call the canonizer {c}(). You passed the class instead of an instance."
#                 )

#         self.zennit_composite = zennit_composite

#     def _iterate_children(self, parent: nn.Module, rule_dict):

#         # for name, child in parent.named_children():

#         #     self._attach_rule(child, parent, name, rule_dict, verbose)
#         #     self._iterate_children(child, rule_dict, verbose)
#         for name, layer in parent.named_modules():
#             xai_module = rule_dict[name](layer)

#             # setattr(layer)

#         return True
