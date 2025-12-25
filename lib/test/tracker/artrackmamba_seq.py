from lib.test.tracker.basetracker import BaseTracker
from lib.models.artrack_mamba_seq import build_artrack_mamba_seq
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box
import torch

class ARTrackMambaSeq(BaseTracker):
    """
    适配 PyTracking 测试框架的 Tracker 类
    """
    def __init__(self, params, dataset_name):
        super(ARTrackMambaSeq, self).__init__(params)
        self.dataset_name = dataset_name
        
        # 1. 构建网络 (使用 params.cfg 中的配置)
        self.network = build_artrack_mamba_seq(self.params.cfg)
        
        # 2. 加载权重
        # 必须处理 'net' 前缀，因为 ltr_trainer 保存时会包一层
        checkpoint = torch.load(self.params.checkpoint, map_location='cpu')
        if 'net' in checkpoint:
            self.network.load_state_dict(checkpoint['net'], strict=True)
        else:
            self.network.load_state_dict(checkpoint, strict=True)
            
        self.network.cuda()
        self.network.eval()
        
        self.preprocessor = Preprocessor() # 需定义或复用 ARTrack 的 Preprocessor
        self.state = None

    def initialize(self, image, info):
        # 初始化状态: xywh
        self.state = info['init_bbox']
        
        # 提取模板特征 (Z)
        # 这里简化逻辑，实际需参考 ARTrack 的 sample_target
        z_patch, _, _ = sample_target(image, self.state, self.params.template_factor, output_sz=self.params.template_size)
        
        # 预处理并转 Tensor
        self.z_dict = {
            'template': self.preprocessor.process(z_patch).cuda()
        }

    def track(self, image, info=None):
        H, W, _ = image.shape
        
        # 1. 提取搜索区域 (X)
        x_patch, resize_factor, _ = sample_target(image, self.state, self.params.search_factor, output_sz=self.params.search_size)
        x_tensor = self.preprocessor.process(x_patch).cuda()
        
        # 2. 推理
        with torch.no_grad():
            # 这里调用我们在 artrack_mamba_seq.py 里统一过的接口 (返回 dict)
            outputs = self.network(self.z_dict['template'], x_tensor)
            
        # 3. 解析结果
        # 假设 pred_boxes 是 [1, 4] 归一化坐标 (cx, cy, w, h)
        pred_box = outputs['pred_boxes'].squeeze(0).cpu().numpy()
        
        # 4. 映射回原图坐标
        # 反归一化 -> 加上 crop 偏移 -> clip
        pred_box = self.map_box_back(pred_box, resize_factor)
        self.state = clip_box(pred_box, H, W, margin=10)

        return {"target_bbox": self.state}

    def map_box_back(self, pred_box, resize_factor):
        # 简单的坐标映射逻辑 (参考 ARTrack 原版实现)
        cx_prev, cy_prev, w_prev, h_prev = self.state
        half_side = 0.5 * self.params.search_size / resize_factor
        
        cx_real = pred_box[0] * (self.params.search_size / resize_factor) + (cx_prev - half_side)
        # ... (略，需补全具体数学计算)
        return [cx_real, cy_real, w_prev, h_prev] # 仅作示例

class Preprocessor(object):
    def process(self, img_arr):
        # Normalize & ToTensor
        img_tensor = torch.tensor(img_arr).permute(2, 0, 1).float().unsqueeze(0)
        return img_tensor