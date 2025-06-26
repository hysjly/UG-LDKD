from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS
from Dynamic.my_encoder_module.preprocessing import Preprocessing

@TRANSFORMS.register_module()
class GenerateCustomMask(BaseTransform):
    """自定义的掩码生成步骤，利用 Preprocessing 生成掩码"""

    def __init__(self, data_path, json_path, **kwargs):
        super().__init__(**kwargs)
        self.preprocessing = Preprocessing(data_path, json_path)

    def __call__(self, results):
        """实现生成掩码的逻辑"""

        # 检查是否需要生成掩码
        if results.get('use_mask', False):
            image_id = results['img_info']['id']
            mask = self.preprocessing.generate_mask(image_id)
            results['gt_masks'] = mask

        return results
