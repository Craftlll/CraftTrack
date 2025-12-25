import torch
from lib.test.tracker.basetracker import BaseTracker
from lib.models.artrackmamba_seq import build_artrack_mamba_seq
from lib.train.data.processing_utils import sample_target
import torch.nn.functional as F

class ARTrackMambaTracker(BaseTracker):
    def __init__(self, params, dataset_name):
        super(ARTrackMambaTracker, self).__init__(params)
        self.dataset_name = dataset_name
        
        # 1. 构建网络
        self.network = build_artrack_mamba_seq(params.cfg)
        self.network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.network.cuda()
        self.network.eval()
        
        # 2. 状态存储 (Memory Bank)
        # self.memory = [] # 用于 RAG

    def initialize(self, image, info):
        """
        第一帧初始化
        """
        # Crop 模板区域
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch = self.preprocessor.process(z_patch_arr, z_amask_arr)
        
        # 保存初始状态
        self.state = info['init_bbox']
        self.frame_id = 0
        
        # Mamba 不需要显式保存 hidden state (如果是非流式推理模式)
        # 但如果是流式推理 (Stateful Mamba)，这里需要 init_state

    def track(self, image, info=None):
        """
        后续帧跟踪
        """
        H, W, _ = image.shape
        self.frame_id += 1
        
        # 1. Crop 搜索区域 (基于上一帧位置 self.state)
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                    output_sz=self.params.search_size)
        x_patch = self.preprocessor.process(x_patch_arr, x_amask_arr)
        
        # 2. 网络推理
        with torch.no_grad():
            # forward 返回 pred_logits, pred_boxes
            logits, boxes = self.network(self.z_patch, x_patch)
            
        # 3. 解码结果
        # boxes: [1, 4, 4] -> 我们取平均或特定 token 的 box
        # 这里假设 reg_head 直接输出了 [cx, cy, w, h] (归一化)
        pred_box = boxes[0].mean(dim=0) # [4]
        
        # 4. 反归一化到原图坐标
        pred_box = (pred_box * self.params.search_size) / resize_factor
        
        # 加上 crop 的偏移量
        cx = pred_box[0] + self.state[0] + (1 - self.params.search_factor) * self.state[2] / 2 # 简化逻辑，需根据 sample_target 调整
        cy = pred_box[1] + self.state[1] + (1 - self.params.search_factor) * self.state[3] / 2
        w = pred_box[2]
        h = pred_box[3]
        
        # 更新状态
        self.state = [cx - w/2, cy - h/2, w, h] # xywh
        
        # 5. (可选) 记忆库更新逻辑
        # if confidence > threshold:
        #    self.memory.append(...)
        
        return {"target_bbox": self.state}

    def _post_process(self, pred_box, resize_factor):
        # 实现具体的坐标还原逻辑
        pass