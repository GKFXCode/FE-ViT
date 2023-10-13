from class_registry import ClassRegistry
RUNNER = ClassRegistry()
from .hr_estimate_runner import HREstimateRunner
from .hr_estimate_multi_task_runner import HREstimateMultiTaskRunner
from .hr_estimate_bvp_runner import HREstimateBVPRunner

def get_runner(cfg):
    return RUNNER.get(cfg.runner.name, cfg)