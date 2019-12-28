from functools import partial
from typing import Any, Callable, Dict, Iterable, List
from torch import nn, optim

from viswsl.config import Config
from viswsl.data.vocabulary import SentencePieceVocabulary
from viswsl.models import WordMaskingModel
from viswsl.modules import visual_stream as vstream, textual_stream as tstream
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


class VisualStreamFactory(Factory):

    PRODUCTS = {
        "blind": vstream.BlindVisualStream,
        "torchvision": vstream.TorchvisionVisualStream,
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

    PRODUCTS = {"default": tstream.DefaultTextualStream}

    @classmethod
    def from_config(cls, config: Config) -> nn.Module:
        _C = config

        vocabulary = SentencePieceVocabulary(_C.DATA.VOCABULARY)
        return cls.create(
            _C.MODEL.TEXTUAL.NAME.split("::")[0],
            vocab_size=len(vocabulary),
            hidden_size=_C.MODEL.TEXTUAL.HIDDEN_SIZE,
            num_attention_heads=_C.MODEL.TEXTUAL.NUM_ATTENTION_HEADS,
            num_layers=_C.MODEL.TEXTUAL.NUM_LAYERS,
            activation=_C.MODEL.TEXTUAL.ACTIVATION,
            dropout=_C.MODEL.TEXTUAL.DROPOUT,
            padding_idx=vocabulary.pad_index,
        )


class FusionFactory(Factory):

    PRODUCTS: Dict[str, Callable[..., fusion.Fusion]] = {
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
            kwargs["num_heads"] = _C.MODEL.FUSION.NUM_ATTENTION_HEADS

        return cls.create(_C.MODEL.FUSION.NAME, **kwargs)


class PretrainingModelFactory(Factory):

    PRODUCTS = {"word_masking": WordMaskingModel}

    @classmethod
    def from_config(cls, config: Config) -> nn.Module:
        _C = config
        visual = VisualStreamFactory.from_config(_C)
        textual = TextualStreamFactory.from_config(_C)
        fusion = FusionFactory.from_config(_C)

        # Form kwargs according to the model name, different models require
        # different sets of kwargs in their constructor.
        kwargs = {}
        return cls.create(_C.MODEL.NAME, visual, textual, fusion, **kwargs)


class OptimizerFactory(Factory):

    PRODUCTS = {"sgd": optim.SGD, "adam": optim.Adam, "adamw": optim.AdamW}

    @classmethod
    def from_config(  # type: ignore
        cls, config: Config, named_parameters: Iterable[Any]
    ) -> optim.Optimizer:
        _C = config

        # No weight decay for some params -- typically norm layers and biases.
        # fmt: off
        decay = [
            param for name, param in named_parameters
            if not any(n in name for n in _C.OPTIM.NO_DECAY)
        ]
        no_decay = [
            param for name, param in named_parameters
            if any(n in name for n in _C.OPTIM.NO_DECAY)
        ]
        # fmt: on
        param_groups = [
            {"params": decay, "weight_decay": _C.OPTIM.WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        # Form kwargs according to the optimizer name, different optimizers
        # may require different hyperparams in their constructor, for example:
        # `SGD` accepts "momentum" while `Adam` doesn't.
        kwargs = {"lr": _C.OPTIM.LR, "weight_decay": _C.OPTIM.WEIGHT_DECAY}
        if _C.OPTIM.OPTIMIZER_NAME == "sgd":
            kwargs["momentum"] = _C.OPTIM.SGD_MOMENTUM
            kwargs["nesterov"] = _C.OPTIM.SGD_NESTEROV

        optimizer = cls.create(_C.OPTIM.OPTIMIZER_NAME, param_groups, **kwargs)
        if _C.OPTIM.USE_LOOKAHEAD:
            optimizer = Lookahead(
                optimizer, k=_C.OPTIM.LOOKAHEAD_STEPS, alpha=_C.OPTIM.LOOKAHEAD_ALPHA
            )
        return optimizer


class LRSchedulerFactory(Factory):

    PRODUCTS = {
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
