from functools import partial
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import albumentations as alb
from torch import nn, optim

from viswsl.config import Config
import viswsl.data as vdata
from viswsl.data import transforms as T
from viswsl.data.tokenizer import SentencePieceBPETokenizer
import viswsl.models as vmodels
from viswsl.modules import visual_stream as vs, textual_stream as ts
from viswsl.optim import Lookahead, lr_scheduler


class Factory(object):

    PRODUCTS: Dict[str, Any] = {}

    def __init__(self):
        raise ValueError(
            f"""Cannot instantiate {self.__class__.__name__} object, use
            `create` classmethod to create a product from this factory.
            """
        )

    @property
    def products(self) -> List[str]:
        return list(self.PRODUCTS.keys())

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> Any:
        if name not in cls.PRODUCTS:
            raise KeyError(f"{cls.__class__.__name__} cannot create {name}.")

        return cls.PRODUCTS[name](*args, **kwargs)

    @classmethod
    def from_config(cls, config: Config) -> Any:
        raise NotImplementedError


class TokenizerFactory(Factory):

    PRODUCTS = {"SentencePieceBPETokenizer": SentencePieceBPETokenizer}

    @classmethod
    def from_config(cls, config: Config) -> SentencePieceBPETokenizer:
        _C = config

        tokenizer = cls.create(
            "SentencePieceBPETokenizer",
            vocab_path=_C.DATA.TOKENIZER_VOCAB,
            model_path=_C.DATA.TOKENIZER_MODEL,
        )
        return tokenizer


class ImageTransformsFactory(Factory):

    # fmt: off
    PRODUCTS = {
        # Input resize transforms: whenever selected, these are always applied.
        # These transforms require one position argument: image dimension.
        "random_resized_crop": partial(
            T.RandomResizedSquareCrop, scale=(0.2, 1.0), ratio=(0.75, 1.333), p=1.0
        ),
        "center_crop": partial(T.CenterSquareCrop, p=1.0),
        "smallest_resize": partial(alb.SmallestMaxSize, p=1.0),
        "global_resize": partial(T.SquareResize, p=1.0),

        # Keep hue limits small in color jitter because it changes color drastically
        # and captions often mention colors. Apply with higher probability.
        "color_jitter": partial(
            T.ColorJitter, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8
        ),
        "horizontal_flip": partial(T.HorizontalFlip, p=0.5),

        # Color normalization: whenever selected, always applied.
        "normalize": partial(
            alb.Normalize, mean=T.IMAGENET_COLOR_MEAN, std=T.IMAGENET_COLOR_STD, p=1.0
        ),
    }
    # fmt: on

    @classmethod
    def from_config(cls, config: Config):
        r"""Augmentations cannot be created from config, only :meth:`create`."""
        raise NotImplementedError


class DatasetFactory(Factory):

    PRODUCTS = {
        "word_masking": vdata.WordMaskingPretextDataset,
        "captioning": vdata.CaptioningDataset,
        "bicaptioning": vdata.CaptioningPretextDataset,
        "token_classification": vdata.CaptioningPretextDataset,
        "instance_classification": vdata.InstanceClassificationDataset,
    }

    @classmethod
    def from_config(
        cls,
        config: Config,
        tokenizer: Optional[SentencePieceBPETokenizer] = None,
        split: str = "train",  # one of {"train", "val"}
    ):
        _C = config
        kwargs = {"data_root": _C.DATA.ROOT, "split": split}

        # Create a list of image transformations based on transform names.
        image_transform_list: List[Callable] = []

        for name in getattr(_C.DATA, f"IMAGE_TRANSFORM_{split.upper()}"):
            # Pass dimensions if cropping / resizing, else rely on the defaults
            # as per `ImageTransformsFactory`.
            if "resize" in name or "crop" in name:
                image_transform_list.append(
                    ImageTransformsFactory.create(name, _C.DATA.IMAGE_CROP_SIZE)
                )
            else:
                image_transform_list.append(ImageTransformsFactory.create(name))

        kwargs["image_transform"] = alb.Compose(image_transform_list)

        # Add model specific kwargs. Refer call signatures of specific datasets.
        if _C.MODEL.NAME != "instance_classification":
            tokenizer = tokenizer or TokenizerFactory.from_config(_C)
            kwargs.update(
                tokenizer=tokenizer,
                max_caption_length=_C.DATA.MAX_CAPTION_LENGTH,
                use_single_caption=_C.DATA.USE_SINGLE_CAPTION,
                percentage=_C.DATA.USE_PERCENTAGE if split == "train" else 100.0,
            )
            if _C.MODEL.NAME == "word_masking":
                kwargs.update(
                    mask_proportion=_C.PRETEXT.WORD_MASKING.MASK_PROPORTION,
                    mask_probability=_C.PRETEXT.WORD_MASKING.MASK_PROBABILITY,
                    replace_probability=_C.PRETEXT.WORD_MASKING.REPLACE_PROBABILITY,
                )

        # Dataset names match with model names (and ofcourse pretext names).
        return cls.create(_C.MODEL.NAME, **kwargs)


class VisualStreamFactory(Factory):

    PRODUCTS = {
        "blind": vs.BlindVisualStream,
        "torchvision": vs.TorchvisionVisualStream,
    }

    @classmethod
    def from_config(cls, config: Config) -> vs.VisualStream:
        _C = config
        kwargs = {"visual_feature_size": _C.MODEL.VISUAL.FEATURE_SIZE}
        if "torchvision" in _C.MODEL.VISUAL.NAME:
            zoo_name, cnn_name = _C.MODEL.VISUAL.NAME.split("::")
            kwargs["pretrained"] = _C.MODEL.VISUAL.PRETRAINED
            kwargs["frozen"] = _C.MODEL.VISUAL.FROZEN

            return cls.create(zoo_name, cnn_name, **kwargs)
        else:
            return cls.create(_C.MODEL.VISUAL.NAME, **kwargs)


class TextualStreamFactory(Factory):

    # fmt: off
    PRODUCTS: Dict[str, Callable] = {
        "allfuse_prenorm": partial(ts.AllLayersFusionTextualStream, norm_type="pre"),
        "allfuse_postnorm": partial(ts.AllLayersFusionTextualStream, norm_type="post"),
        "none": None,  # Keep for pretext tasks which don't use captions.
    }
    # fmt: on

    @classmethod
    def from_config(
        cls, config: Config, tokenizer: Optional[SentencePieceBPETokenizer] = None
    ) -> Union[None, nn.Module]:

        _C = config
        name = _C.MODEL.TEXTUAL.NAME.split("::")[0]
        if name == "none":
            return None

        tokenizer = tokenizer or TokenizerFactory.from_config(_C)
        # Transformer will be bidirectional only for word masking pretext.
        kwargs = {
            "vocab_size": tokenizer.get_vocab_size(),
            "hidden_size": _C.MODEL.TEXTUAL.HIDDEN_SIZE,
            "dropout": _C.MODEL.DROPOUT,
            "is_bidirectional": _C.MODEL.NAME == "word_masking",
            "padding_idx": tokenizer.token_to_id("[UNK]"),
            "max_caption_length": _C.DATA.MAX_CAPTION_LENGTH,
            "feedforward_size": _C.MODEL.TEXTUAL.FEEDFORWARD_SIZE,
            "attention_heads": _C.MODEL.TEXTUAL.ATTENTION_HEADS,
            "num_layers": _C.MODEL.TEXTUAL.NUM_LAYERS,
        }
        return cls.create(name, **kwargs)


class PretrainingModelFactory(Factory):

    PRODUCTS = {
        "word_masking": vmodels.WordMaskingModel,
        "captioning": partial(vmodels.CaptioningModel, is_bidirectional=False),
        "bicaptioning": partial(vmodels.CaptioningModel, is_bidirectional=True),
        "token_classification": vmodels.TokenClassificationModel,
        "instance_classification": vmodels.InstanceClassificationModel,
    }

    @classmethod
    def from_config(
        cls, config: Config, tokenizer: Optional[SentencePieceBPETokenizer] = None
    ) -> nn.Module:

        _C = config
        tokenizer = tokenizer or TokenizerFactory.from_config(_C)

        visual = VisualStreamFactory.from_config(_C)
        textual = TextualStreamFactory.from_config(_C, tokenizer)

        # Check textual stream being none for fixed set of pretext tasks.
        if textual is None:
            assert _C.MODEL.NAME in {
                "token_classification",
                "instance_classification",
            }, f"Textual stream can't be none for {_C.MODEL.NAME}"

        # Add model specific kwargs. Refer call signatures of specific models
        # for matching kwargs here.
        kwargs = {}
        if _C.MODEL.NAME == "captioning":
            kwargs.update(
                max_decoding_steps=_C.DATA.MAX_CAPTION_LENGTH,
                sos_index=tokenizer.token_to_id("[SOS]"),
                eos_index=tokenizer.token_to_id("[EOS]"),
            )

        elif _C.MODEL.NAME == "token_classification":
            kwargs.update(
                vocab_size=tokenizer.get_vocab_size(),
                ignore_indices=[
                    tokenizer.token_to_id("[UNK]"),
                    tokenizer.token_to_id("[SOS]"),
                    tokenizer.token_to_id("[EOS]"),
                    tokenizer.token_to_id("[MASK]"),
                ],
            )
        elif _C.MODEL.NAME == "instance_classification":
            kwargs.update(
                vocab_size=81,  # 80 COCO categories + background (padding, 0)
                ignore_indices=[0],  # background index
            )

        if textual is not None:
            return cls.create(_C.MODEL.NAME, visual, textual, **kwargs)
        else:
            return cls.create(_C.MODEL.NAME, visual, **kwargs)


class OptimizerFactory(Factory):

    PRODUCTS = {"sgd": optim.SGD}

    @classmethod
    def from_config(
        cls, config: Config, named_parameters: Iterable[Any]
    ) -> optim.Optimizer:
        _C = config

        # Set different learning rate for CNN and rest of the model during
        # pretraining. This doesn't matter for downstream evaluation because
        # there are no modules with "cnn" in their name.
        # Also turn off weight decay for layer norm and bias in textual stream.
        param_groups = []
        for name, param in named_parameters:
            wd = 0.0 if re.match(_C.OPTIM.NO_DECAY, name) else _C.OPTIM.WEIGHT_DECAY
            lr = _C.OPTIM.CNN_LR if "cnn" in name else _C.OPTIM.LR
            param_groups.append({"params": [param], "lr": lr, "weight_decay": wd})

        kwargs = {"momentum": _C.OPTIM.SGD_MOMENTUM}
        optimizer = cls.create(_C.OPTIM.OPTIMIZER_NAME, param_groups, **kwargs)

        if _C.OPTIM.USE_LOOKAHEAD:
            optimizer = Lookahead(
                optimizer, k=_C.OPTIM.LOOKAHEAD_STEPS, alpha=_C.OPTIM.LOOKAHEAD_ALPHA
            )
        return optimizer


class LRSchedulerFactory(Factory):

    PRODUCTS = {
        "none": lr_scheduler.LinearWarmupNoDecayLR,
        "multistep": lr_scheduler.LinearWarmupMultiStepLR,
        "linear": lr_scheduler.LinearWarmupLinearDecayLR,
        "cosine": lr_scheduler.LinearWarmupCosineAnnealingLR,
    }

    @classmethod
    def from_config(
        cls, config: Config, optimizer: optim.Optimizer
    ) -> optim.lr_scheduler.LambdaLR:
        _C = config

        kwargs = {"warmup_steps": _C.OPTIM.WARMUP_STEPS}

        # Multistep LR requires multiplicative factor and milestones.
        if _C.OPTIM.LR_DECAY_NAME == "multistep":
            kwargs["gamma"] = _C.OPTIM.LR_GAMMA
            kwargs["milestones"] = _C.OPTIM.LR_STEPS
        else:
            kwargs["total_steps"] = _C.OPTIM.NUM_ITERATIONS

        return cls.create(_C.OPTIM.LR_DECAY_NAME, optimizer, **kwargs)


class DownstreamDatasetFactory(Factory):
    # Key names correspond to `DATA.ROOT` of downstream config.
    PRODUCTS = {
        "datasets/VOC2007": vdata.VOC07ClassificationDataset,
        "datasets/imagenet": vdata.ImageNetDataset,
    }

    @classmethod
    def from_config(cls, config: Config, split: str = "train"):
        _C = config
        kwargs = {"data_root": _C.DATA.ROOT, "split": split}

        image_transform_names: List[str] = list(
            _C.DATA.IMAGE_TRANSFORM_TRAIN
            if "train" in split
            else _C.DATA.IMAGE_TRANSFORM_VAL
        )
        # Create a list of image transformations based on names.
        image_transform_list: List[Callable] = []

        for name in image_transform_names:
            # Pass dimensions if cropping / resizing, else rely on the defaults
            # as per `ImageTransformsFactory`.
            if name in {"random_resized_crop", "center_crop", "global_resize"}:
                transform = ImageTransformsFactory.create(name, 224)
            elif name in {"smallest_resize"}:
                transform = ImageTransformsFactory.create(name, 256)
            else:
                transform = ImageTransformsFactory.create(name)
            image_transform_list.append(transform)

        kwargs["image_transform"] = alb.Compose(image_transform_list)

        return cls.create(_C.DATA.ROOT, **kwargs)
