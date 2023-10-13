from class_registry import ClassRegistry
MODEL = ClassRegistry()
from .cnn3d import *
from .vit import *
from .autoformer import *
from .resnet import *
from .ts_can import *
def get_model(cfg):
    return MODEL.get(cfg.model.name, cfg)
