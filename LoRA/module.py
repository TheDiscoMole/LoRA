import math
import torch

class ConvNd (torch.nn.Module):
    """
    ConvNd module for LoRA tasks

    dims: number of input dimensions
    module: base module
    rank: function to calculate layer rank                                      (optional|default: sqrt of smallest dimension)
    """
    def __init__ (self, dims, module,
        rank=lambda *features: int(math.sqrt(min(*features))),
    ):
        torch.nn.Module.__init__(self)

        # configs
        rank = rank(module.in_channels, module.out_channels)

        # validation
        assert rank <= min(module.in_channels, module.out_channels), "rank must be less than smallest dimension"

        # parameters
        self.A = torch.nn.Parameter(torch.empty(*module.kernel_size, module.out_channels, rank))
        self.B = torch.nn.Parameter(torch.empty(*module.kernel_size, rank, module.in_channels // module.groups))

        self.reset_parameters()

        # convolution
        match dims:
            case 1: self.convolution = torch.nn.functional.conv1d
            case 2: self.convolution = torch.nn.functional.conv2d
            case 3: self.convolution = torch.nn.functional.conv3d

        # module
        self.get_base_module = lambda: module

    def forward (self, input):
        return self.convolution(
            input=input,
            weight=self.get_base_module().weight + (self.A @ self.B).view(self.get_base_module().weight.shape),
            bias=self.get_base_module().bias,
            stride=self.get_base_module().stride,
            padding=self.get_base_module().padding,
            dilation=self.get_base_module().dilation,
            groups=self.get_base_module().groups)

    def reset_parameters (self):
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

class Embedding (torch.nn.Module):
    """
    Embedding module for LoRA tasks

    module: base module
    rank: function to calculate layer rank                                      (optional|default: sqrt of smallest dimension)
    """
    def __init__ (self, module,
        rank=lambda *features: int(math.sqrt(min(*features))),
    ):
        torch.nn.Module.__init__(self)

        # configs
        rank = rank(module.num_embeddings, module.embedding_dim)

        # validation
        assert rank <= min(module.num_embeddings, module.embedding_dim), "rank must be less than smallest dimension"

        # configs
        self.num_embeddings = module.num_embeddings
        self.embedding_dim = module.embedding_dim

        # parameters
        self.A = torch.nn.Parameter(torch.empty(module.num_embeddings, rank))
        self.B = torch.nn.Parameter(torch.empty(rank, module.embedding_dim))

        self.reset_parameters()

        # module
        self.get_base_module = lambda: module

    def forward (self, input):
        return torch.nn.functional.embedding(
            input=input,
            weight=self.get_base_module().weight + (self.A @ self.B),
            padding_idx=self.get_base_module().padding_idx,
            max_norm=self.get_base_module().max_norm,
            norm_type=self.get_base_module().norm_type,
            scale_grad_by_freq=self.get_base_module().scale_grad_by_freq,
            sparse=self.get_base_module().sparse)

    def reset_parameters (self):
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

class Linear (torch.nn.Module):
    """
    Linear module for LoRA tasks

    module: base module
    rank: function to calculate layer rank                                      (optional|default: sqrt of smallest dimension)
    """
    def __init__ (self, module,
        rank=lambda *features: int(math.sqrt(min(*features))),
    ):
        torch.nn.Module.__init__(self)

        # configs
        rank = rank(module.in_features, module.out_features)

        # validation
        assert rank <= min(module.in_features, module.out_features), "rank must be less than smallest dimension"

        # configs
        self.in_features = module.in_features
        self.out_features = module.out_features

        # parameters
        self.A = torch.nn.Parameter(torch.empty(module.out_features, rank))
        self.B = torch.nn.Parameter(torch.empty(rank, module.in_features))

        self.reset_parameters()

        # module
        self.get_base_module = lambda: module

    def forward (self, input):
        return torch.nn.functional.linear(
            input=input,
            weight=self.get_base_module().weight + (self.A @ self.B),
            bias=self.get_base_module().bias)

    def reset_parameters (self):
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

base_module_wrappers = {}

def register_base_module_wrapper (base_type, task_contructor_func):
    base_module_wrappers[base_type] = task_contructor_func

def get_module_task_constructor (module):
    for base_type, task_contructor_func in base_module_wrappers.items():
        if isinstance(module, base_type): return task_contructor_func

register_base_module_wrapper(torch.nn.Conv1d, lambda module: ConvNd(dims=1, module=module))
register_base_module_wrapper(torch.nn.Conv2d, lambda module: ConvNd(dims=2, module=module))
register_base_module_wrapper(torch.nn.Conv3d, lambda module: ConvNd(dims=3, module=module))
register_base_module_wrapper(torch.nn.Embedding, lambda module: Embedding(module=module))
register_base_module_wrapper(torch.nn.Linear, lambda module: Linear(module=module))
