import collections
import torch

from .module import *

class Layer (torch.nn.Module):
    """
    LoRA module wrapper layer

    model: lora model reference
    module: module to lora wrap
    new_task: lambda to generate new task
    """
    def __init__ (self, model, module, new_task):
        torch.nn.Module.__init__(self)

        # validation
        assert isinstance(model, Model), "model must be LoRA model"
        assert not isinstance(module, Layer), "module cannot be LoRA layer"

        # configs
        self.new_task = new_task

        # module
        self.module = module
        self.tasks  = torch.nn.ModuleDict()

        # model
        self.get_model = lambda: model

    def forward (self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs) if self.get_model().task is None else self.tasks[self.get_model().task](*inputs, **kwargs)

    def add_task (self, task):
        self.tasks[task] = self.new_task(self.module)
        return self

    def remove_task (self, task):
        del self.tasks[task]
        return self

    def state_dict (self, prefix="", task=None, **kwargs):
        return self.module.state_dict(prefix=f"{prefix}module.", **kwargs) if task is None else self.tasks[task].state_dict(prefix=f"{prefix}module.", **kwargs)

    def load_state_dict (self, state_dict, prefix="", task=None, **kwargs):
        if task is None: return self.module.load_state_dict(state_dict, **kwargs)

        prefix = f"{prefix}module."
        state_dict = {k[len(prefix)]:v for k,v in state_dict.items() if prefix in k}
        self.tasks[task].load_state_dict(state_dict, **kwargs)

        return self

class Model (torch.nn.Module):
    """
    LoRA model wrapper

    model: base model
    """
    def __init__ (self, model,
        **kwargs
    ):
        torch.nn.Module.__init__(self)

        # validation
        assert not isinstance(model, Model), "model is already LoRA model"

        # compiler
        def compile (model):
            assert not isinstance(model, Layer), "model already contains LoRA layers"

            for name, module in model.named_children():
                constructor = get_module_task_constructor(module)

                if constructor is None: compile(module)
                else: setattr(model, name, Layer(self, module, constructor))
        compile(model)

        # model
        self.model = model

    def forward (self, *inputs, task=None, **kwargs):
        self.task = task
        output = self.model(*inputs, **kwargs)
        self.task = None

        return output

    def lora_layers (self):
        for module in self.modules():
            if isinstance(module, Layer): yield module

    def add_task (self, task):
        for module in self.lora_layers(): module.add_task(task=task)
        return self

    def remove_task (self, task):
        for module in self.lora_layers(): module.remove_task(task=task)
        return self

    def state_dict (self, prefix="", task=None, **kwargs):
        if task is None: return self.model.state_dict(destination=collections.OrderedDict(), prefix=prefix)

        def df_task_state_dict (module, prefix, task):
            task_state_dict = {}

            if isinstance(module, Layer):
                return module.state_dict(prefix=f"{prefix}{task}.", task=task)

            for name,child in module.named_children():
                task_state_dict = {**task_state_dict, **df_task_state_dict(child, f"{prefix}{name}.", task)}

            return task_state_dict

        return collections.OrderedDict(df_task_state_dict(self.model, prefix, task))

    def load_state_dict (self, state_dict, prefix="", task=None, **kwargs):
        if task is None: return self.model.load_state_dict(state_dict)

        def df_load_task_state_dict (module, state_dict, prefix, task):
            if isinstance(module, Layer):
                return module.load_state_dict(state_dict, prefix=f"{prefix}{task}.", task=task)

            for name,child in module.named_children():
                df_load_task_state_dict (child, state_dict, f"{prefix}{name}.", task)
        df_load_task_state_dict(self.model, state_dict, prefix, task)

        return self
