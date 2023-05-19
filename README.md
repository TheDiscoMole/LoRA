# LoRA: Low-Rank Adaptation

This repo contains the source code of the Python package `LoRA` and serves as re-implementation of `loralib`

## Motivation

This re-implementation serves as nothing more than a less invasive, more dynamic and *seemingly* `torch`-native restructuring of the `loralib` functionality. By turning LoRA modules into compiled and unstructured `Module` wrappers we can achieve the following quality of life:

1. There is no need to rewrite models with custom LoRA modules.
2. We can store and train/infer over multiple different tasks at once. <br>
*scroll down to **why multiple task** for some of the reasoning behind this*

Hopefully this implementation helps some of you out there (as it makes out of the box fine-tuning a little easier) or serve as some inspiration for `loralib`.

*this repo is "stable", but in production you are on your own*

## Paper and Authors

**LoRA: Low-Rank Adaptation of Large Language Models** <br>
*Edward J. Hu\*, Yelong Shen\*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen* <br>

>LoRA reduces the number of trainable parameters by learning pairs of rank-decompostion matrices while freezing the original weights.
This vastly reduces the storage requirement for large language models adapted to specific tasks and enables efficient task-switching during deployment all without introducing inference latency.
LoRA also outperforms several other adaptation methods including adapter, prefix-tuning, and fine-tuning.

Library: [loralib](https://github.com/microsoft/LoRA)<br>
Paper: [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685) <br>
Citation: <br>
```
@misc{hu2021lora,
    title={LoRA: Low-Rank Adaptation of Large Language Models},
    author={Hu, Edward and Shen, Yelong and Wallis, Phil and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Lu and Chen, Weizhu},
    year={2021},
    eprint={2106.09685},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## Quickstart

1. Installing `LoRA`
```
pip install git+https://github.com/TheDiscoMole/LoRA
```

2. Write your model and wrap it up in LoRA goodness. <br>
*`LoRA.Model` alters the computational graph, so be sure to load your base model checkpoint before this step if necessary*
```
import LoRA

model = Diffusion_Model()       # your model
model = LoRA.Model(model=model) # your model with LoRA
```

3. Add/Remove LoRA tasks from your model.
```
model.add_task("minimalist") # diffusion LoRA task for a minimalist art style
model.add_task("anime")      # diffusion LoRA task for an anime art style
model.remove_task("anime")   # because weebs are scum
```

4. Freeze parameters if you like.
```
model.requires_grad_(requires_grad=False)                    # freezes the base model parameters
model.requires_grad_(requires_grad=False, task="minimalist") # freezes LoRA task model parameters
```

5. When computing outputs during training or inference specify your LoRA task.
```
model(input)                    # model outputs without LoRA task
model(input, task="minimalist") # model outputs with LoRA task
```

6. When saving a checkpoint using `state_dict`, specify your LoRA task.
```
checkpoint = model.state_dict()                  # get base model parameters
checkpoint = model.state_dict(task="minimalist") # get LoRA task parameters ONLY
```

7. When loading a checkpoint using `load_state_dict`, specify your LoRA task.
```
model.load_state_dict(checkpoint)                    # set base model parameters
model.load_state_dict(checkpoint, task="minimalist") # set LoRA task parameters ONLY
```

This library was designed to appear as `torch` native, and be as syntactically non-invasive as possible.

## Custom LoRA Module

This re-implementation natively supports the following base modules:

1. `torch.nn.Linear`
2. `torch.nn.Embedding`
3. `torch.nn.ConvNd` *(N=1,2,3)*

To add your own module type that you want `LoRA.Model` to wrap, write your LoRA module and pass the base module with a LoRA module constructor function to `LoRA.register_base_module_wrapper` like so:
```
# your custom definition for how to wrap transformers
class LoRATransformer (torch.nn.Module):
	def __init__ (self, module, *args, **kwargs):
        ...

LoRA.register_base_module_wrapper(torch.nn.Transformer, lambda module: LoRATransformer(module=module, *scoped_args, **scoped_kwargs))
```

This registers the passed `lambda` function as a task constructor for your custom `LoRATransformer` when `LoRA.Model()` encounters a `torch.nn.Transformer` in the computational graph.<br>
**Note**: `LoRA.Model()` traverses the computational graph lazily, so once it encounters a `torch.nn.Module` to wrap it ignores that module's sub-graph.

## why multiple task

My personal research projects often revolve around multi-modal and reusable graphs and sub-graphs. Having the ability to interleave task specific training batches, instead of reloading the LoRA `state_dict` every task epoch, is both convenient and results in a more stable and rapidly converging model.

The next step would be to implement the handling of multiple tasks simultaneously. This could be used to achieve some more modest task training granularity:
```
LoRADiffusion(prompt, tasks=["surrealism","pokemon"])
```

or be used to fragment a model's computational graph entirely: (instead of embedding a classifiable feature space, fragment the network along the class spaces)
```
LoRADiffusion(prompt, tasks=["surrealism","cubeism","expressionism","birds","horses","trees","landscape"])
```
## Contributing

This repository mainly serves personal research purposes. Contributions are welcome, but might be better directed at `loralib`.

This repository uses the `MIT License`.
