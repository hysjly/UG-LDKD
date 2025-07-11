# Copyright (c) OpenMMLab. All rights reserved.
"""MMRazor provides 17 registry nodes to support using modules across projects.
Each node is a child of the root registry in MMEngine.

More details can be found at
https://mmengine.readthedocs.io/en/latest/tutorials/registry.html.
"""
from typing import Any, Dict, Optional, Union

from mmengine.config import Config, ConfigDict
from mmrazor.registry import DATA_SAMPLERS as MMENGINE_DATA_SAMPLERS
from mmrazor.registry import DATASETS as MMENGINE_DATASETS
from mmrazor.registry import HOOKS as MMENGINE_HOOKS
from mmrazor.registry import LOOPS as MMENGINE_LOOPS
from mmrazor.registry import METRICS as MMENGINE_METRICS
from mmrazor.registry import MODEL_WRAPPERS as MMENGINE_MODEL_WRAPPERS
from mmrazor.registry import MODELS as MMENGINE_MODELS
from mmrazor.registry import \
    OPTIM_WRAPPER_CONSTRUCTORS as MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS
from mmrazor.registry import OPTIM_WRAPPERS as MMENGINE_OPTIM_WRAPPERS
from mmrazor.registry import OPTIMIZERS as MMENGINE_OPTIMIZERS
from mmrazor.registry import PARAM_SCHEDULERS as MMENGINE_PARAM_SCHEDULERS
from mmrazor.registry import \
    RUNNER_CONSTRUCTORS as MMENGINE_RUNNER_CONSTRUCTORS
from mmrazor.registry import RUNNERS as MMENGINE_RUNNERS
from mmrazor.registry import TASK_UTILS as MMENGINE_TASK_UTILS
from mmrazor.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmrazor.registry import VISBACKENDS as MMENGINE_VISBACKENDS
from mmrazor.registry import VISUALIZERS as MMENGINE_VISUALIZERS
from mmrazor.registry import \
    WEIGHT_INITIALIZERS as MMENGINE_WEIGHT_INITIALIZERS
from mmengine.registry import Registry, build_from_cfg


# pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
# mmdet 3.3.0 requires mmcv<2.2.0,>=2.0.0rc4; extra == "mim", but you have mmcv 2.2.0 which is incompatible.
def build_razor_model_from_cfg(
        cfg: Union[dict, ConfigDict, Config],
        registry: 'Registry',
        default_args: Optional[Union[dict, ConfigDict, Config]] = None) -> Any:
    # TODO relay on mmengine:HAOCHENYE/config_new_feature
    if cfg.get('cfg_path', None) and not cfg.get('type', None):
        from mmengine.hub import get_model
        model = get_model(**cfg)  # type: ignore
        return model

    return_architecture = False
    if cfg.get('_return_architecture_', None):
        return_architecture = cfg.pop('_return_architecture_')
    razor_model = build_from_cfg(cfg, registry, default_args)
    if return_architecture:
        return razor_model.architecture
    else:
        return razor_model


# Registries For Runner and the related
# manage all kinds of runners like `EpochBasedRunner` and `IterBasedRunner`
RUNNERS = Registry('runner', parent=MMENGINE_RUNNERS)
# manage runner constructors that define how to initialize runners
RUNNER_CONSTRUCTORS = Registry(
    'runner constructor', parent=MMENGINE_RUNNER_CONSTRUCTORS)
# manage all kinds of loops like `EpochBasedTrainLoop`
LOOPS = Registry('loop', parent=MMENGINE_LOOPS)
# manage all kinds of hooks like `CheckpointHook`
HOOKS = Registry('hook', parent=MMENGINE_HOOKS)

# Registries For Data and the related
# manage data-related modules
DATASETS = Registry('dataset', parent=MMENGINE_DATASETS)
DATA_SAMPLERS = Registry('data sampler', parent=MMENGINE_DATA_SAMPLERS)
TRANSFORMS = Registry('transform', parent=MMENGINE_TRANSFORMS)

# manage all kinds of modules inheriting `nn.Module`
MODELS = Registry(
    'model', parent=MMENGINE_MODELS, build_func=build_razor_model_from_cfg, locations=['razor.models'])
# manage all kinds of model wrappers like 'MMDistributedDataParallel'
MODEL_WRAPPERS = Registry('model_wrapper', parent=MMENGINE_MODEL_WRAPPERS)
# manage all kinds of weight initialization modules like `Uniform`
WEIGHT_INITIALIZERS = Registry(
    'weight initializer', parent=MMENGINE_WEIGHT_INITIALIZERS)

# Registries For Optimizer and the related
# manage all kinds of optimizers like `SGD` and `Adam`
OPTIMIZERS = Registry('optimizer', parent=MMENGINE_OPTIMIZERS)
# manage optimizer wrapper
OPTIM_WRAPPERS = Registry('optimizer_wrapper', parent=MMENGINE_OPTIM_WRAPPERS)
# manage constructors that customize the optimization hyperparameters.
OPTIM_WRAPPER_CONSTRUCTORS = Registry(
    'optimizer wrapper constructor',
    parent=MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS)
# manage all kinds of parameter schedulers like `MultiStepLR`
PARAM_SCHEDULERS = Registry(
    'parameter scheduler', parent=MMENGINE_PARAM_SCHEDULERS)

# manage all kinds of metrics
METRICS = Registry('metric', parent=MMENGINE_METRICS)

# manage task-specific modules like anchor generators and box coders
TASK_UTILS = Registry('task util', parent=MMENGINE_TASK_UTILS)

# Registries For Visualizer and the related
# manage visualizer
VISUALIZERS = Registry('visualizer', parent=MMENGINE_VISUALIZERS)
# manage visualizer backend
VISBACKENDS = Registry('vis_backend', parent=MMENGINE_VISBACKENDS)


# manage sub models for downstream repos
@MODELS.register_module()
def sub_model(cfg,
              fix_subnet,
              mode: str = 'mutable',
              prefix: str = '',
              extra_prefix: str = '',
              init_weight_from_supernet: bool = False,
              init_cfg: Optional[Dict] = None,
              **kwargs):
    model = MODELS.build(cfg)
    # Save path type cfg process, set init_cfg directly.
    if init_cfg:
        # update init_cfg when init_cfg is valid.
        model.init_cfg = init_cfg

    if init_weight_from_supernet:
        # init weights from supernet first before it turns into a sub model.
        model.init_weights()

    from mmrazor.structures import load_fix_subnet

    load_fix_subnet(
        model,
        fix_subnet,
        load_subnet_mode=mode,
        prefix=prefix,
        extra_prefix=extra_prefix)

    if not init_weight_from_supernet:
        # init weights from the specific sub model.
        model.init_weights()

    return model
