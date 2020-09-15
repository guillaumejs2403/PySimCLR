import os
import yaml
import argparse

from core.step_functions import simclr_step
from core.optimization import optimization, linear_training

from models.cifar.model_generator import SimCLR_Model

import torch


def arguments():
    parser = argparse.ArgumentParser(description='SimCLR routine')
    parser.add_argument('-c', '--config', default='config-cifar.yaml',
                        type=str, help='Yaml config file')
    parser.add_argument('-g', '--gpu-id', default='2', type=str,
                        help='GPU id')

    return parser.parse_args()


if __name__ == '__main__':
    args = arguments()

    # ============================================================================
    # Config loading and CUDA settings
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device("cuda:0")

    # ============================================================================
    # Model loading
    model = SimCLR_Model(config['model'])
    model.load_state_dict(
        torch.load('{}/checkpoint.pth'.format(config['output_dir']),
                   map_location='cpu')['model']
    )
    model.to(device)
    model.eval()

    # ============================================================================
    # load step function and optimization
    step_function = simclr_step(config['loss']['temperature'], device)
    optim_utils = optimization()

    # ============================================================================
    # perform evaluation
    top_epoch, best_train_acc, best_acc = linear_training(model=model,
                                                          device=device,
                                                          step_class=step_function,
                                                          config=config,
                                                          optim_utils=optim_utils)

    with open('{}/linear_eval.txt'.format(config['output_dir']), 'w') as f:
        f.write('Epoch: {}\nTrain acc: {}\nTest acc: {}'.format(
            top_epoch, best_train_acc, best_acc
        ))
