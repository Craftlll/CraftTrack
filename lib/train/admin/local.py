class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/root/CraftTrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/root/CraftTrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/root/CraftTrack/pretrained_networks'
        self.lasot_dir = '/root/autodl-tmp/data/lasot'
        self.got10k_dir = '/root/autodl-tmp/data/got10k/train'
        self.got10k_val_dir = '/root/autodl-tmp/data/got10k/val'
        self.lasot_lmdb_dir = '/root/autodl-tmp/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/root/autodl-tmp/data/got10k_lmdb'
        self.trackingnet_dir = '/root/autodl-tmp/data/trackingnet'
        self.trackingnet_lmdb_dir = '/root/autodl-tmp/data/trackingnet_lmdb'
        self.coco_dir = '/root/autodl-tmp/data/coco'
        self.coco_lmdb_dir = '/root/autodl-tmp/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/root/autodl-tmp/data/vid'
        self.imagenet_lmdb_dir = '/root/autodl-tmp/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
