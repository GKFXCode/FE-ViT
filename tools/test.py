from vhwz_video.runner import get_runner
import hydra
from omegaconf import OmegaConf
import sys
import datetime
import logging

@hydra.main(config_path='../config')
def test(cfg):
    print(OmegaConf.to_yaml(cfg))
    
    runner = get_runner(cfg)
    model = runner.get_model()
    res = runner.eval(model)
    logging.info(res)
    
    
if __name__ == '__main__':
    print('start testing')
    test()