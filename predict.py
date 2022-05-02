import cog
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import argparse
from virtex.config import Config
from virtex.data import ImageDirectoryDataset
from virtex.factories import TokenizerFactory, PretrainingModelFactory
from virtex.utils.checkpointing import CheckpointManager
from collections import defaultdict
from typing import Callable, Dict, List, Any
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from virtex.data import transforms as T


class Predictor(cog.Predictor):
    def setup(self):
        self.config = 'configs/width_ablations/bicaptioning_R_50_L1_H2048.yaml'
        self.checkpoint_path = 'models/bicaptioning_R_50_L1_H2048.pth'
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    @cog.input("image", type=Path, help="image for caption generation")
    def predict(self, image):
        caption = self.gen_caption(image)
        return caption

    def gen_caption(self, image_path):
        _C = Config(self.config, [])
        tokenizer = TokenizerFactory.from_config(_C)
        val_dataloader = DataLoader(
            SingleImageDataset(image_path),
            batch_size=_C.OPTIM.BATCH_SIZE,
            num_workers=4,
            pin_memory=True,
        )

        model = PretrainingModelFactory.from_config(_C)
        if torch.cuda.is_available():
            model.to(self.device)
        ITERATION = CheckpointManager(model=model).load(self.checkpoint_path)
        model.eval()

        # Make a list of predictions to evaluate.
        predictions: List[Dict[str, Any]] = []

        for val_iteration, val_batch in enumerate(val_dataloader, start=1):
            if torch.cuda.is_available():
                val_batch["image"] = val_batch["image"].to(self.device)
            with torch.no_grad():
                output_dict = model(val_batch)

            # Make a dictionary of predictions in COCO format.
            for image_id, caption in zip(
                    val_batch["image_id"], output_dict["predictions"]
            ):
                predictions.append(
                    {
                        # Convert image id to int if possible (mainly for COCO eval).
                        "image_id": int(image_id) if image_id.isdigit() else image_id,
                        "caption": tokenizer.decode(caption.tolist()),
                    }
                )

        pred = predictions[0]
        return pred['caption']



class SingleImageDataset(Dataset):

    def __init__(
        self, image_path: str, image_transform: Callable = T.DEFAULT_IMAGE_TRANSFORM
    ):
        self.image_path = image_path
        self.image_transform = image_transform

    def __len__(self):
        return 1

    def __getitem__(self, idx: int):
        # Open image from path and apply transformation, convert to CHW format.
        image = cv2.imread(str(self.image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.image_transform(image=image)["image"]
        image = np.transpose(image, (2, 0, 1))
        return {"image_id": str(idx), "image": torch.tensor(image)}
