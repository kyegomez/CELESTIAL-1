
import copy
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from cstx.dataset.utils import process_caption


class BaseDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
            self, 
            data_path: str,
            mm_root_path: str, 
            embed_path: str
        ):
        super(BaseDataset, self).__init__()
        self.embed_path = embed_path
        self.mm_path_list, self.caption_list = [], []

    def __len__(self):  # number of instances
        return len(self.mm_path_list)

    def __getitem__(self, i):
        with open(os.path.join(
            self.embed_path, str(os.path.basename(self.mm_path_list[i])) + '.npy'
        ), 'rb') as f:
            caption_embs = torch.from_numpy(np.load(f, allow_pickle=True))  # (num_clip_tokens, 768)

        return dict(mm_paths=self.mm_path_list[i], output_texts=self.caption_list[i], caption_embs=caption_embs)

    def collate(self, instances):
        mm_paths, output_texts, caption_embs = tuple(
            [instance[key] for instance in instances] for key in (
                "mm_paths", 
                "output_texts", 
                "caption_embs")
            )
        return dict(
            mm_paths=mm_paths,
            output_texts=output_texts,
            caption_embs=caption_embs
        )

