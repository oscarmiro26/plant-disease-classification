from ..utils.resnet_utils import load_model
from collections import OrderedDict
from operator import add
from torch import nn, randn
from torch.ao.pruning._experimental.pruner import FPGMPruner
from pprint import pprint
from torch.nn.utils import parametrize
from .custom_pruning_func import (
    prune_conv2d_bn2d_relu_conv2d,
    prune_conv2d_bn2d_add_relu_conv2d,
    prune_conv2d_bn2d_ds_add_relu_conv2d,
    prune_conv2d_bn2d_add_relu_pool,
    prune_conv2d_bn2d_conv2d,
    prune_conv2d_bn2d
)


def main():
    model = load_model()
    dummy = randn(1, 3, 224, 224)

    #––– Inspect the original model
    # print("\ORIGINAL MODEL:")
    # for name, m in model.named_modules():
    #     print(f"  {name}: {type(m)}")

    # ── RECORD SHAPES BEFORE PRUNING ──
    shapes_before = {}
    for name, m in model.named_modules():
        print(f"  {name}: {type(m)}")
        if isinstance(m, nn.Conv2d):
            shapes_before[f"{name}.weight"] = tuple(m.weight.shape)
            if m.bias is not None:
                shapes_before[f"{name}.bias"]   = tuple(m.bias.shape)
        elif isinstance(m, nn.BatchNorm2d):
            shapes_before[f"{name}.weight"] = tuple(m.weight.shape)
            if m.bias is not None:
                shapes_before[f"{name}.bias"]   = tuple(m.bias.shape)

    #––– Build pruning config for Conv2d layers
    config = []
    for name, m in model.named_modules():
        if name == 'conv1':
            continue
        if isinstance(m, (nn.Conv2d)):
            fq = f"{name}.weight"
            # print(f"Name: {name}, weight.shape: {m.weight.shape}")
            config.append({"tensor_fqn": fq, "sparsity_level": 0.5})
    # print("\nFINAL CONFIG:")
    # pprint(config)

    #––– Apply FPGMPruner and then inspect parametrizations
    pruner = FPGMPruner(sparsity_level=0.5)

    pruner.patterns.clear()

    resnet_patterns = {
        (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Conv2d): prune_conv2d_bn2d_relu_conv2d,
        (nn.Conv2d, nn.BatchNorm2d, nn.Conv2d): prune_conv2d_bn2d_conv2d,
    }
        
    pruner.patterns.update(resnet_patterns)
    pruner.prepare(model, config)
    pruner.step()
    pruner.enable_mask_update = True
    model = pruner.prune()

    # ── RECORD SHAPES AFTER PRUNING ──
    shapes_after = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            shapes_after[f"{name}.weight"] = tuple(m.weight.shape)
            if m.bias is not None:
                shapes_after[f"{name}.bias"]   = tuple(m.bias.shape)
        elif isinstance(m, nn.BatchNorm2d):
            shapes_after[f"{name}.weight"] = tuple(m.weight.shape)
            if m.bias is not None:
                shapes_after[f"{name}.bias"]   = tuple(m.bias.shape)

    # ── PRINT COMPARISON ──
    print("\nShape changes after pruning:")
    for key in sorted(set(shapes_before) | set(shapes_after)):
        before = shapes_before.get(key)
        after  = shapes_after.get(key)
        # format the tuples as strings
        print(f"{key:40s}  before={before!s:15}  after={after!s}")

    # #--- Inspect the pruned model
    # print("\nPRUNED MODEL:")
    # for name, m in model.named_modules():
    #     if isinstance(m, nn.Conv2d) and 'downsample' not in name:
    #         print(f"  {name}: out_channels={m.out_channels}, weight.shape={m.weight.shape}")
    #     elif isinstance(m, nn.BatchNorm2d) and 'downsample' not in name:
    #         print(f"  {name}: num_features={m.num_features}, running_mean.shape={m.running_mean.shape}")
            

    #--- Check if the model can still run

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            m.register_forward_pre_hook(
                lambda module, inp, name=name:
                    print(f"[Conv2d] {name}: input={tuple(inp[0].shape)}, weight={tuple(module.weight.shape)}")
            )
        elif isinstance(m, nn.BatchNorm2d):
            m.register_forward_pre_hook(
                lambda module, inp, name=name:
                    print(f"[BatchNorm2d] {name}: input={tuple(inp[0].shape)}, weight={tuple(module.weight.shape)}")
            )

    try:
        output = model(dummy)
        print("Model ran successfully after pruning.")
        print("Output shape:", output.shape)
    except Exception as e:
        print("Model failed to run after pruning:", e)
if __name__ == "__main__":
    main()

