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

    def state_dict (self, task=None, prefix="", **kwargs):
        state_dict = {}

        if task is None: self.module._save_to_state_dict(destination=state_dict, prefix=f"{prefix}module.", keep_vars=False)
        else: return self.tasks[task].state_dict(prefix=f"{prefix}module.")

        return state_dict

    def load_state_dict (self, state_dict, task=None, strict=True, **kwargs):
        if task is None: return self.module.load_state_dict({k[len("module."):]:v for k,v in state_dict.items() if "module." in k}, strict=strict)
        else: return self.tasks[task].load_state_dict({k[len(f"{task}.module."):]:v for k,v in state_dict.items() if f"{task}.module." in k}, strict=strict)

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

    def state_dict (self, task=None, **kwargs):

        def df_state_dict (module, prefix, task):
            state_dict = {}

            if isinstance(module, Layer): return module.state_dict(prefix=prefix if task is None else f"{prefix}{task}.", task=task)
            elif task is None: module._save_to_state_dict(destination=state_dict, prefix=prefix, keep_vars=False)

            for name,child in module.named_children():
                state_dict = {**state_dict, **df_state_dict(child, f"{prefix}{name}.", task)}

            return state_dict

        return collections.OrderedDict(df_state_dict(self.model, "", task))

    def load_state_dict (self, state_dict, task=None, strict=True, **kwargs):
        missing_keys = []

        def df_load_task_state_dict (module, state_dict, task, missing_keys):
            if isinstance(module, Layer): return missing_keys.extend(module.load_state_dict(state_dict, task=task, strict=strict).missing_keys)
            elif task is None: module._load_from_state_dict(state_dict, "", {}, strict, missing_keys, [], [])

            for name,child in module.named_children():
                df_load_task_state_dict (child, {k[len(f"{name}."):]:v for k,v in state_dict.items() if f"{name}." in k}, task, missing_keys)
        df_load_task_state_dict(self.model, state_dict, task, missing_keys)

        return torch.nn.modules.module._IncompatibleKeys(missing_keys=missing_keys, unexpected_keys=[])
