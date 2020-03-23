from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional

import albumentations as alb
from torch import nn, optim

from viswsl.config import Config
import viswsl.data as vdata
from viswsl.data.tokenizer import SentencePieceBPETokenizer
from viswsl.data.transforms import IMAGENET_COLOR_MEAN, IMAGENET_COLOR_STD
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
        tokenizer = tokenizer or TokenizerFactory.from_config(_C)

        # Add model specific kwargs. Refer call signatures of specific datasets.
        # TODO (kd): InstanceClassificationDataset does not accept most of the
        # args. Make the API more consistent.
        if _C.MODEL.NAME != "instance_classification":
            kwargs = {
                "lmdb_path": _C.DATA.VAL_LMDB
                if split == "val"
                else _C.DATA.TRAIN_LMDB,
                "tokenizer": tokenizer,
                "max_caption_length": _C.DATA.MAX_CAPTION_LENGTH,
                "use_single_caption": _C.DATA.USE_SINGLE_CAPTION,
                "percentage": _C.DATA.USE_PERCENTAGE if split == "train" else 100.0,
            }
            if _C.MODEL.NAME == "word_masking":
                kwargs.update(
                    mask_proportion=_C.PRETEXT.WORD_MASKING.MASK_PROPORTION,
                    mask_probability=_C.PRETEXT.WORD_MASKING.MASK_PROBABILITY,
                    replace_probability=_C.PRETEXT.WORD_MASKING.REPLACE_PROBABILITY,
                )
        else:
            # TODO: add `root` argument after adding to config.
            kwargs = {
                "shuffle": split == "train",
                "split": split,
            }

        # Set random flip as image only or image-caption based on pretext task.
        HorizontalFlip = (
            vdata.transforms.ImageCaptionHorizontalFlip
            if _C.MODEL.NAME != "instance_classification"
            else alb.HorizontalFlip
        )
        # Prepare a list of augmentations based on split (train or val).
        if split == "train":
            augmentation_list: List[Callable] = [
                alb.RandomResizedCrop(
                    _C.DATA.IMAGE_CROP_SIZE,
                    _C.DATA.IMAGE_CROP_SIZE,
                    scale=(0.2, 1.0),
                    ratio=(0.75, 1.333),
                    always_apply=True,
                ),
                HorizontalFlip(p=0.5),
                alb.RandomBrightnessContrast(
                    brightness_limit=0.4, contrast_limit=0.4, p=0.5
                ),
                alb.HueSaturationValue(
                    hue_shift_limit=40, sat_shift_limit=40, val_shift_limit=0, p=0.5
                ),
            ]
        else:
            augmentation_list = [
                alb.SmallestMaxSize(
                    max_size=_C.DATA.IMAGE_CROP_SIZE, always_apply=True
                ),
                alb.CenterCrop(
                    _C.DATA.IMAGE_CROP_SIZE,
                    _C.DATA.IMAGE_CROP_SIZE,
                    always_apply=True,
                ),
            ]

        augmentation_list.append(alb.ToFloat(max_value=255.0))
        augmentation_list.append(
            alb.Normalize(
                mean=IMAGENET_COLOR_MEAN, std=IMAGENET_COLOR_STD, max_pixel_value=1.0
            )
        )

        kwargs["image_transform"] = alb.Compose(augmentation_list)
        # Dataset names match with model names (and ofcourse pretext names).
        return cls.create(_C.MODEL.NAME, **kwargs)


class DownstreamDatasetFactory(Factory):
    # We use `DOWNSTREAM.LINEAR_CLF.DATA_ROOT` so these keys look like paths.
    PRODUCTS = {
        "datasets/imagenet": vdata.ImageNetDataset,
        "datasets/places205": vdata.Places205Dataset,
    }

    @classmethod
    def from_config(cls, config: Config, split: str = "train"):
        _C = config
        kwargs = {"root": _C.DOWNSTREAM.LINEAR_CLF.DATA_ROOT, "split": split}
        return cls.create(_C.DOWNSTREAM.LINEAR_CLF.DATA_ROOT, **kwargs)


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
        return cls.create(_C.MODEL.VISUAL.NAME, **kwargs)


class TextualStreamFactory(Factory):

    # fmt: off
    PRODUCTS: Dict[str, Callable] = {
        "allfuse_prenorm": partial(ts.AllLayersFusionTextualStream, norm_type="pre"),
        "allfuse_postnorm": partial(ts.AllLayersFusionTextualStream, norm_type="post"),
    }
    # fmt: on

    @classmethod
    def from_config(
        cls, config: Config, tokenizer: Optional[SentencePieceBPETokenizer] = None
    ) -> nn.Module:

        _C = config
        name = _C.MODEL.TEXTUAL.NAME.split("::")[0]
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
        # Let the default values in `instance_classification` do the job right
        # now. Change them later.

        return cls.create(_C.MODEL.NAME, visual, textual, **kwargs)


class OptimizerFactory(Factory):

    PRODUCTS = {"sgd": optim.SGD, "adam": optim.Adam, "adamw": optim.AdamW}

    @classmethod
    def from_config(  # type: ignore
        cls, config: Config, named_parameters: Iterable[Any]
    ) -> optim.Optimizer:
        _C = config

        # Form param groups on two criterions:
        #   1. no weight decay for some parameters (usually norm and bias)
        #   2. different LR and weight decay for CNN and rest of model.
        # fmt: off
        param_groups: List[Dict[str, Any]] = []
        for name, param in named_parameters:
            lr = _C.OPTIM.CNN_LR if "cnn" in name else _C.OPTIM.LR

            is_no_decay = any(n in name for n in _C.OPTIM.NO_DECAY)
            wd = 0.0 if is_no_decay else _C.OPTIM.WEIGHT_DECAY

            param_groups.append({"params": [param], "lr": lr, "weight_decay": wd})
        # fmt: on

        if "adam" in _C.OPTIM.OPTIMIZER_NAME:
            kwargs = {"betas": tuple(_C.OPTIM.ADAM_BETAS)}
        else:
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
        "linear": lr_scheduler.LinearWarmupLinearDecayLR,
        "cosine": lr_scheduler.LinearWarmupCosineAnnealingLR,
    }

    @classmethod
    def from_config(  # type: ignore
        cls, config: Config, optimizer: optim.Optimizer
    ) -> optim.lr_scheduler.LambdaLR:
        _C = config
        return cls.create(
            _C.OPTIM.LR_DECAY_NAME,
            optimizer,
            total_steps=_C.OPTIM.NUM_ITERATIONS,
            warmup_steps=_C.OPTIM.WARMUP_STEPS,
        )
