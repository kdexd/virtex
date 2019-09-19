import math
import os
import random
from typing import List

import dataflow as df
import lmdb
import numpy as np
import torch
from torch.utils.data import get_worker_info, IterableDataset

from viswsl.data.tokenizers import SentencePieceTokenizer
from viswsl.data.vocabulary import SentencePieceVocabulary
from viswsl.utils.pretraining import mask_some_tokens_randomly


# TODO (kd): Write a class for val split with sequential read, no shuffle.
class CocoCaptionsTrainDataset(IterableDataset):
    def __init__(
        self,
        lmdb_path: str,
        vocabulary: SentencePieceVocabulary,
        tokenizer: SentencePieceTokenizer,
        max_caption_length: int = 25,
        buffer_size: int = 8,
    ):

        self._lmdb_path = lmdb_path
        self._vocabulary = vocabulary
        self._tokenizer = tokenizer
        self._max_caption_length = max_caption_length
        self._buffer_size = buffer_size

        # Get a list of "keys" in the LMDB file so we could shard the dataset
        # according to the number of worker processes.
        with lmdb.open(
            self._lmdb_path,
            subdir=os.path.isdir(self._lmdb_path),
            readonly=True,
            lock=False,
            readahead=True,
            map_size=1099511627776 * 2,
        ) as _lmdb_file:

            _txn = _lmdb_file.begin()
            self._keys: List[bytes] = df.utils.serialize.loads(
                _txn.get(b"__keys__")
            )

        # List of augmentations to be applied on each image after reading
        # from LMDB. This follows the standard augmentation steps of
        # (ImageNet pre-trained) ResNet models:
        #     1. Resize shortest edge to 256. (Already done in LMDB)
        #     2. Convert pixel intensities in [0, 1].
        #     3. Random crop a (224, 224) patch.
        #     4. Normalize image by mean pixel intensity and variance.
        #     5. Convert from HWC to CHW format.
        self._image_augmentor = df.imgaug.AugmentorList(
            [
                df.imgaug.RandomCrop(224),
                df.imgaug.ToFloat32(),
                df.imgaug.MapImage(lambda image: image / 255.0),
                df.imgaug.MapImage(
                    lambda image: (image - np.array([0.485, 0.456, 0.406]))
                    / np.array([0.229, 0.224, 0.225])
                ),
                df.imgaug.MapImage(
                    lambda image: np.transpose(image, (2, 0, 1))
                ),
            ]
        )

    def __len__(self):
        return len(self._keys)

    def __iter__(self):

        # ====================================================================
        # This code block will be executed just once, when an iterator of this
        # dataset is intialized: ``iter(dataset)`` or ``enumerate(dataset)``.
        # --------------------------------------------------------------------
        # Shard the dataset according to the number of workers.
        # Two members of interest:
        #     1. ``id: int = [0, n-1]``
        #     2. ``num_workers: int = n``.
        worker_info = get_worker_info()

        if worker_info is not None:
            _per_worker = int(
                math.ceil(len(self._keys) / worker_info.num_workers)
            )
            start = _per_worker * worker_info.id

            # Last worker may get less than ``_per_worker`` examples.
            end = min(len(self._keys), _per_worker * (worker_info.id + 1))
        else:
            # Single process data-loading, don't shard the dataset.
            start = 0
            end = len(self._keys)

        # Load examples from serialized LMDB (sharded by number of workers).
        # Read sequentially, random reads on large datasets may be expensive.
        pipeline = df.LMDBData(
            self._lmdb_path, keys=self._keys[start:end], shuffle=False
        )

        # Decode bytes read from LMDB to Python objects.
        pipeline = df.MapData(pipeline, df.LMDBSerializer._deserialize_lmdb)

        # Keep a fixed-size buffer: examples will be pushed in this buffer and
        # randomly selected to make batches; a good proxy for random reads.
        pipeline = df.LocallyShuffleData(pipeline, self._buffer_size)
        pipeline.reset_state()
        # ====================================================================

        for instance in pipeline:
            image, captions = instance

            image = self._image_augmentor.augment(image)

            # Select a caption randomly (during training).
            caption = random.choice(captions)

            # Tokenize and trim to max length - 2 (count [CLS] and [SEP]).
            caption_tokens = self._tokenizer.tokenize(caption)
            caption_tokens = caption_tokens[: self._max_caption_length - 2]

            # Add [CLS] and [SEP] tokens. [SEP] is simply EOS, or </S> token.
            caption_tokens.insert(0, self._vocabulary.cls_token)
            caption_tokens.append(self._vocabulary.sep_token)

            # Pad the sequence of tokens up to maximum length.
            # This makes the default ``collate_fn`` of dataloader work.
            caption_tokens.extend(
                [self._vocabulary.pad_token]
                * (self._max_caption_length - len(caption_tokens))
            )
            # Mask out few tokens randomly.
            caption_tokens, masked_labels = mask_some_tokens_randomly(
                caption_tokens,
                mask_token=self._vocabulary.mask_token,
                pad_token=self._vocabulary.pad_token,
                ignore_tokens=[
                    self._vocabulary.cls_token,
                    self._vocabulary.sep_token,
                ],
            )
            # Convert (string) tokens to (integer) token indices.
            token_indices = [
                self._vocabulary.get_token_index(t) for t in caption_tokens
            ]
            masked_label_indices = [
                self._vocabulary.get_token_index(t) for t in masked_labels
            ]

            yield {
                "image": torch.tensor(image),
                "tokens": torch.tensor(token_indices).long(),
                "masked_labels": torch.tensor(masked_label_indices).long(),
            }
