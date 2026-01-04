import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os

class LasotTestDataset(BaseDataset):
    """
    LaSOT Test dataset (flattened structure)
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.lasottest_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        # Flattened structure: no class_name subfolder
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        occlusion_label_path = '{}/{}/full_occlusion.txt'.format(self.base_path, sequence_name)

        # NOTE: pandas backed seems super super slow for loading occlusion/oov masks
        full_occlusion = load_text(str(occlusion_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        out_of_view_label_path = '{}/{}/out_of_view.txt'.format(self.base_path, sequence_name)
        out_of_view = load_text(str(out_of_view_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        target_visible = np.logical_and(full_occlusion == 0, out_of_view == 0)

        frames_path = '{}/{}/img'.format(self.base_path, sequence_name)

        frames_list = ['{}/{:08d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]

        target_class = sequence_name.split('-')[0]
        return Sequence(sequence_name, frames_list, 'lasottest', ground_truth_rect.reshape(-1, 4),
                        object_class=target_class, target_visible=target_visible)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        # Dynamically list directories in base_path that look like sequences
        # Filter for directories that contain groundtruth.txt to be safe
        seq_list = []
        if not os.path.exists(self.base_path):
            print("Warning: {} does not exist".format(self.base_path))
            return []

        for d in os.listdir(self.base_path):
            dir_path = os.path.join(self.base_path, d)
            if os.path.isdir(dir_path):
                # Check for groundtruth.txt
                if os.path.exists(os.path.join(dir_path, 'groundtruth.txt')):
                    seq_list.append(d)
        
        return sorted(seq_list)
