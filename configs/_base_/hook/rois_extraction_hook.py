from mmengine.hooks import Hook
from mmengine.runner import Runner

class RoIsExtractionHook(Hook):
    def __init__(self, target_module_name='roi_head'):
        self.target_module_name = target_module_name

    def before_train_iter(self, runner: Runner, batch_idx, data_batch=None):
        # 获取模型的 roi_head 模块
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        target_module = getattr(model, self.target_module_name, None)
        if target_module is None:
            raise ValueError(f'Module {self.target_module_name} not found in the model.')

        # 注册 forward hook 到 _mask_forward 函数以提取 RoIs
        def hook_fn(module, input, output):
            # 获取 RoIs，通常是 _mask_forward 中的第二个参数
            rois = input[1]  # input 是一个 tuple，rois 通常为第二个元素
            runner.outputs['mask_rois'] = rois

        # 在 _mask_forward 函数上注册 hook
        self.hook_handle = target_module._mask_forward.register_forward_hook(hook_fn)

    def after_train_iter(self, runner: Runner, batch_idx, data_batch=None, outputs=None):
        # 每次迭代结束后移除 hook，防止多次迭代产生重复 hook
        if hasattr(self, 'hook_handle'):
            self.hook_handle.remove()
            del self.hook_handle

# # 在您的配置文件中添加该自定义 Hook
# custom_hooks = [
#     dict(type='RoIsExtractionHook')
# ]
