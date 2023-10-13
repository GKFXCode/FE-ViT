from vhwz_video.runner import get_runner
import hydra
from omegaconf import OmegaConf
import sys
import datetime

@hydra.main(config_path='../config')
def train(cfg):
    print(OmegaConf.to_yaml(cfg))
    
    runner = get_runner(cfg)
    runner.train()
    
    
if __name__ == '__main__':
    print('start training')
    train()