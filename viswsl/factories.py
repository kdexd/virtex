from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional

import albumentations as alb
import tokenizers as tkz
from torch import nn, optim

from viswsl.config import Config
import viswsl.data as vdata
import viswsl.models as vmodels
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
        # and boundary token (SOS/EOS).
        special_tokens = ["[UNK]", "[MASK]", "[B]"]

        # Add a leading space only for SentencePiece.
        kwargs: Dict[str, Any] = {}
        if _C.DATA.CAPTION.TOKENIZER == "SentencePieceBPETokenizer":
            kwargs = {"add_prefix_space": True, "unk_token": "[UNK]"}

        tokenizer = cls.create(_C.DATA.CAPTION.TOKENIZER, **kwargs)

        # Train tokenizer on given caption corpus. This will be determisitic
        # for a fixed corpus, vocab size and tokenizer.
        tokenizer.train(
            files=_C.DATA.CAPTION_CORPUS,
            vocab_size=_C.DATA.CAPTION.VOCAB_SIZE,
            special_tokens=special_tokens,
        )
        # Tokenizers from huggingface provide support to handle truncation and
        # padding up to maximum length, but we do it outside in our dataset
        # class for better control (for example, we flip caption for backward
        # captioning and it requires different side of truncation).
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
    def from_config(cls, config: Config) -> nn.Module:
        _C = config
        if "torchvision" in _C.MODEL.VISUAL.NAME:
            cnn_name = _C.MODEL.VISUAL.NAME.split("::")[-1]
            kwargs = {"pretrained": _C.MODEL.VISUAL.PRETRAINED}
            return cls.create("torchvision", cnn_name, **kwargs)

        return cls.create(_C.MODEL.VISUAL.NAME)


class TextualStreamFactory(Factory):

    PRODUCTS: Dict[str, Callable[..., ts.TransformerTextualStream]] = {
        "postnorm_gelu": partial(
            ts.TransformerTextualStream, norm_type="post", activation="gelu"
        ),
        "postnorm_relu": partial(
            ts.TransformerTextualStream, norm_type="post", activation="relu"
        ),
        "prenorm_gelu": partial(
            ts.TransformerTextualStream, norm_type="pre", activation="gelu"
        ),
        "prenorm_relu": partial(
            ts.TransformerTextualStream, norm_type="pre", activation="relu"
        ),
    }

    @classmethod
    def from_config(
        cls,
        config: Config,
        tokenizer: Optional[tkz.implementations.BaseTokenizer] = None,
    ) -> nn.Module:

        _C = config
        tokenizer = tokenizer or TokenizerFactory.from_config(_C)
        kwargs = {
            "vocab_size": _C.DATA.CAPTION.VOCAB_SIZE,
            "hidden_size": _C.MODEL.TEXTUAL.HIDDEN_SIZE,
            "dropout": _C.MODEL.TEXTUAL.DROPOUT,
            "padding_idx": tokenizer.token_to_id("[UNK]"),
        }
        if _C.MODEL.TEXTUAL.NAME != "embedding":
            # Transformer will be bidirectional only for word masking pretext.
            is_bidirectional = _C.MODEL.NAME == "word_masking"
            kwargs.update(
                feedforward_size=_C.MODEL.TEXTUAL.FEEDFORWARD_SIZE,
                attention_heads=_C.MODEL.TEXTUAL.ATTENTION_HEADS,
                num_layers=_C.MODEL.TEXTUAL.NUM_LAYERS,
                is_bidirectional=is_bidirectional,
            )

        return cls.create(_C.MODEL.TEXTUAL.NAME.split("::")[0], **kwargs)


class FusionFactory(Factory):

    PRODUCTS: Dict[str, Callable[..., fusion.Fusion]] = {
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
        kwargs = {
            "visual_feature_size": _C.MODEL.VISUAL.FEATURE_SIZE,
            "textual_feature_size": _C.MODEL.TEXTUAL.HIDDEN_SIZE,
            "projection_size": _C.MODEL.FUSION.PROJECTION_SIZE,
            "dropout": _C.MODEL.FUSION.DROPOUT,
        }
        if _C.MODEL.FUSION.NAME == "multihead":
            kwargs["attention_heads"] = _C.MODEL.FUSION.ATTENTION_HEADS

        return cls.create(_C.MODEL.FUSION.NAME, **kwargs)


class PretrainingModelFactory(Factory):

    PRODUCTS = {
        "word_masking": vmodels.WordMaskingModel,
        "captioning": partial(vmodels.CaptioningModel, bidirectional=False),
        "bicaptioning": partial(vmodels.CaptioningModel, bidirectional=True),
    }

    @classmethod
    def from_config(cls, config: Config) -> nn.Module:
        _C = config
        visual = VisualStreamFactory.from_config(_C)
        textual = TextualStreamFactory.from_config(_C)
        fusion = FusionFactory.from_config(_C)

        # Form kwargs according to the model name, different models require
        # different sets of kwargs in their constructor.
        kwargs = {"tie_embeddings": _C.MODEL.TIE_EMBEDDINGS}
        return cls.create(_C.MODEL.NAME, visual, textual, fusion, **kwargs)


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
            wd = (
                _C.OPTIM.CNN_WEIGHT_DECAY if "cnn" in name else _C.OPTIM.WEIGHT_DECAY
            )
            if any(n in name for n in _C.OPTIM.NO_DECAY):
                wd = 0.0
            param_groups.append({"params": [param], "lr": lr, "weight_decay": wd})
        # fmt: on

        # Form kwargs according to the optimizer name, different optimizers
        # may require different hyperparams in their constructor, for example:
        # `SGD` accepts "momentum" while `Adam` doesn't.
        if "sgd" in _C.OPTIM.OPTIMIZER_NAME:
            kwargs = {
                "momentum": _C.OPTIM.SGD_MOMENTUM,
                "nesterov": _C.OPTIM.SGD_NESTEROV,
            }
        elif "adam" in _C.OPTIM.OPTIMIZER_NAME:
            kwargs = {"betas": (_C.OPTIM.ADAM_BETA1, _C.OPTIM.ADAM_BETA2)}

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
