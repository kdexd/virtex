from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional

import albumentations as alb
import tokenizers as tkz
from torch import nn, optim

from viswsl.config import Config
import viswsl.data as vdata
import viswsl.models as vmodels
from viswsl.modules.embedding import WordAndPositionalEmbedding
from viswsl.modules import visual_stream as vs, textual_stream as ts
from viswsl.modules import fusion
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
    PRODUCTS = {
        "SentencePieceBPETokenizer": tkz.SentencePieceBPETokenizer,
        "ByteLevelBPETokenizer": tkz.ByteLevelBPETokenizer,
    }

    @classmethod
    def from_config(cls, config: Config) -> tkz.implementations.BaseTokenizer:
        _C = config

        # Special tokens: padding/out-of-vocabulary token ([UNK]), mask token,
        # and boundary tokens (SOS/EOS).
        special_tokens = ["[UNK]", "[SOS]", "[EOS]", "[MASK]"]

        # Add a leading space only for SentencePiece.
        kwargs: Dict[str, Any] = {"add_prefix_space": True}
        if _C.DATA.CAPTION.TOKENIZER == "SentencePieceBPETokenizer":
            kwargs["unk_token"] = "[UNK]"

        tokenizer = cls.create(_C.DATA.CAPTION.TOKENIZER, **kwargs)

        # Train tokenizer on given caption corpus. This will be determisitic
        # for a fixed corpus, vocab size and tokenizer.
        tokenizer.train(
            files=_C.DATA.CAPTION_CORPUS,
            vocab_size=_C.DATA.CAPTION.VOCAB_SIZE,
            special_tokens=special_tokens,
            split_by_unicode_script=True,
        )
        return tokenizer


class DatasetFactory(Factory):
    PRODUCTS = {
        "word_masking": vdata.WordMaskingDataset,
        "captioning": vdata.CaptioningDataset,
        "bicaptioning": vdata.CaptioningDataset,
    }

    @classmethod
    def from_config(
        cls,
        config: Config,
        tokenizer: Optional[tkz.implementations.BaseTokenizer] = None,
        split: str = "train",  # one of {"train", "val"}
    ):
        _C = config
        tokenizer = tokenizer or TokenizerFactory.from_config(_C)

        kwargs = {
            "lmdb_path": _C.DATA.VAL_LMDB if split == "val" else _C.DATA.TRAIN_LMDB,
            "tokenizer": tokenizer,
            "random_horizontal_flip": _C.DATA.IMAGE.RANDOM_FLIP,
            "max_caption_length": _C.DATA.CAPTION.MAX_LENGTH,
            "shuffle": False if split == "val" else True,
        }
        if _C.MODEL.NAME == "word_masking":
            kwargs.update(
                mask_proportion=_C.PRETEXT.WORD_MASKING.MASK_PROPORTION,
                mask_probability=_C.PRETEXT.WORD_MASKING.MASK_PROBABILITY,
                replace_probability=_C.PRETEXT.WORD_MASKING.REPLACE_PROBABILITY,
            )

        # Add data augmentations to `image_transform`.
        augmentation_list = [
            alb.SmallestMaxSize(max_size=_C.DATA.IMAGE.RESIZE_SIZE),
            alb.RandomResizedCrop(
                _C.DATA.IMAGE.CROP_SIZE,
                _C.DATA.IMAGE.CROP_SIZE,
                scale=(0.08, 1.0),
                ratio=(0.75, 1.333),
            ),
        ]
        photometric_augmentation = [
            alb.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            alb.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20
            ),
            vdata.AlexNetPCA(),
        ]
        if _C.DATA.IMAGE.PHOTOMETRIC_AUG:
            augmentation_list.extend(photometric_augmentation)

        augmentation_list.append(alb.ToFloat(max_value=255.0))
        if _C.DATA.IMAGE.COLOR_NORMALIZE:
            augmentation_list.append(
                alb.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=1.0,
                )
            )
        kwargs["image_transform"] = alb.Compose(augmentation_list)
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

            return cls.create(zoo_name, cnn_name, **kwargs)
        return cls.create(_C.MODEL.VISUAL.NAME, **kwargs)


class TextualStreamFactory(Factory):

    PRODUCTS: Dict[str, Callable] = {
        "memless_postnorm": partial(ts.MemorylessTextualStream, norm_type="post"),
        "memless_prenorm": partial(ts.MemorylessTextualStream, norm_type="pre"),
        "vismem_postnorm": partial(ts.VisualMemoryTextualStream, norm_type="post"),
        "vismem_prenorm": partial(ts.VisualMemoryTextualStream, norm_type="pre"),
    }

    @classmethod
    def from_config(
        cls,
        config: Config,
        tokenizer: Optional[tkz.implementations.BaseTokenizer] = None,
    ) -> nn.Module:

        _C = config
        name = _C.MODEL.TEXTUAL.NAME.split("::")[0]
        tokenizer = tokenizer or TokenizerFactory.from_config(_C)

        # Transformer will be bidirectional only for word masking pretext.
        kwargs = {
            "vocab_size": tokenizer.get_vocab_size(),
            "hidden_size": _C.MODEL.TEXTUAL.HIDDEN_SIZE,
            "dropout": _C.MODEL.DROPOUT,
            "do_early_fusion": _C.MODEL.TEXTUAL.DO_EARLY_FUSION,
            "is_bidirectional": _C.MODEL.NAME == "word_masking",
            "padding_idx": tokenizer.token_to_id("[UNK]"),
        }
        if _C.MODEL.TEXTUAL.NAME != "embedding":
            kwargs.update(
                feedforward_size=_C.MODEL.TEXTUAL.FEEDFORWARD_SIZE,
                attention_heads=_C.MODEL.TEXTUAL.ATTENTION_HEADS,
                num_layers=_C.MODEL.TEXTUAL.NUM_LAYERS,
            )

        return cls.create(name, **kwargs)


class LateFusionFactory(Factory):

    PRODUCTS: Dict[str, Callable] = {
        "none": fusion.NoFusion,
        "concatenate": fusion.ConcatenateFusion,
        "additive": partial(fusion.ElementwiseFusion, operation="additive"),
        "multiplicative": partial(
            fusion.ElementwiseFusion, operation="multiplicative"
        ),
        "multihead": fusion.MultiheadAttentionFusion,
    }

    @classmethod
    def from_config(cls, config: Config) -> fusion.Fusion:
        _C = config
        # We will project image features to `TEXTUAL.HIDDEN_SIZE` in model.
        kwargs = {
            "feature_size": _C.MODEL.TEXTUAL.HIDDEN_SIZE,
            "dropout": _C.MODEL.DROPOUT,
        }
        if _C.MODEL.LATE_FUSION.NAME == "multihead":
            kwargs["attention_heads"] = _C.MODEL.LATE_FUSION.ATTENTION_HEADS

        return cls.create(_C.MODEL.LATE_FUSION.NAME, **kwargs)


class PretrainingModelFactory(Factory):

    PRODUCTS = {
        "word_masking": vmodels.WordMaskingModel,
        "captioning": partial(vmodels.CaptioningModel, is_bidirectional=False),
        "bicaptioning": partial(vmodels.CaptioningModel, is_bidirectional=True),
    }

    @classmethod
    def from_config(cls, config: Config) -> nn.Module:
        _C = config

        visual = VisualStreamFactory.from_config(_C)
        textual = TextualStreamFactory.from_config(_C)
        late_fusion = LateFusionFactory.from_config(_C)

        return cls.create(_C.MODEL.NAME, visual, textual, late_fusion)


class OptimizerFactory(Factory):

    PRODUCTS = {"adam": optim.Adam, "adamw": optim.AdamW}

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
            wd = (
                _C.OPTIM.CNN_WEIGHT_DECAY if "cnn" in name else _C.OPTIM.WEIGHT_DECAY
            )
            if any(n in name for n in _C.OPTIM.NO_DECAY):
                wd = 0.0
            param_groups.append({"params": [param], "lr": lr, "weight_decay": wd})
        # fmt: on

        optimizer = cls.create(
            _C.OPTIM.OPTIMIZER_NAME, param_groups, betas=tuple(_C.OPTIM.ADAM_BETAS)
        )
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
