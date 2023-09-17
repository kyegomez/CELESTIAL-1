import json
import os

from tqdm import tqdm

from cstx.dataset.base_dataset import BaseDataset
from cstx.dataset.utils import process_caption


class CC3MDataset(BaseDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
            self, 
            data_path: str, 
            mm_root_path: str, 
            embed_path: str
        ):
        super(CC3MDataset, self).__init__(data_path, mm_root_path, embed_path)
        self.embed_path = embed_path

        print('Load CC3M dataset ...')
        self.mm_path_list, self.caption_list = [], []
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for row in tqdm(data, total=len(data)):
            image_id, one_caption = row["image_name"], row["caption"]
            self.mm_path_list.append(os.path.join(mm_root_path, image_id))
            self.caption_list.append(process_caption(one_caption))

        print(f'[!] collect {len(self.mm_path_list)} samples for training')

