from typing import Any, Dict, Iterable, List, Type
from torch import nn, optim

from viswsl.config import Config
from viswsl.data.vocabulary import SentencePieceVocabulary
from viswsl.modules import visual_stream as vstream, textual_stream as tstream
from viswsl.optim import lr_scheduler


class Factory(object):

    PRODUCTS: Dict[str, Type[Any]] = {}

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
    def from_config(cls, config: Config, *args, **kwargs) -> Any:
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
            padding_idx=vocabulary.pad_index,
        )


class OptimizerFactory(Factory):

    PRODUCTS = {"sgd": optim.SGD, "adam": optim.Adam, "adamw": optim.AdamW}

    @classmethod
    def from_config(
        cls, config: Config, named_parameters: Iterable[Any]
    ) -> optim.Optimizer:
        _C = config

        if len(_C.OPTIM.NO_DECAY) > 0:
            # Turn off weight decay for a few params -- typically norm layers
            # and biases.
            # fmt: off
            param_groups = [
                {
                    "params": [
                        param for name, param in named_parameters
                        if not any(nd in name for nd in _C.OPTIM.NO_DECAY)
                    ],
                    "weight_decay": _C.OPTIM.WEIGHT_DECAY
                }, {
                    "params": [
                        param for name, param in named_parameters
                        if any(nd in name for nd in _C.OPTIM.NO_DECAY)
                    ],
                    "weight_decay": 0.0
                }
            ]
            # fmt: on
        else:
            # Apply weight decay to all params equally.
            param_groups = named_parameters  # type: ignore

        # Form kwargs according to the optimizer name, different optimizers
        # may require different hyperparams in their constructor, for example:
        # `SGD` accepts "momentum" while `Adam` doesn't.
        kwargs = {"lr": _C.OPTIM.LR, "weight_decay": _C.OPTIM.WEIGHT_DECAY}
        if _C.OPTIM.OPTIMIZER_NAME == "sgd":
            kwargs["momentum"] = _C.OPTIM.SGD_MOMENTUM
            kwargs["nesterov"] = _C.OPTIM.SGD_NESTEROV

        return cls.create(_C.OPTIM.OPTIMIZER_NAME, param_groups, **kwargs)


class LRSchedulerFactory(Factory):

    PRODUCTS = {
        "linear": lr_scheduler.LinearWarmupLinearDecayLR,
        "cosine": lr_scheduler.LinearWarmupCosineAnnealingLR,
    }

    @classmethod
    def from_config(
        cls, config: Config, optimizer: optim.Optimizer
    ) -> optim.lr_scheduler.LambdaLR:
        _C = config
        return cls.create(
            _C.OPTIM.LR_DECAY_NAME,
            optimizer,
            total_steps=_C.OPTIM.NUM_ITERATIONS,
            warmup_steps=_C.OPTIM.WARMUP_STEPS,
        )
