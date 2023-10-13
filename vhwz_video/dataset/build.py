from class_registry import ClassRegistry
DATASET = ClassRegistry()

def get_dataset(cfg, mode):
    cfg.dataset.mode = mode
    return DATASET.get(cfg.dataset.train.name, cfg)
