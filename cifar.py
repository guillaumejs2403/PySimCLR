import os
import yaml
import argparse
import pandas as pd

import torch
import torch.optim as optim

# custom imports
from core.misc import n_parameters
from core.datasets import get_unsupervised_dataset
from core.optimization import optimization, linear_training
from core.step_functions import simclr_step

from models.cifar.model_generator import SimCLR_Model


def arguments():
    parser = argparse.ArgumentParser(description='SimCLR routine')
    parser.add_argument('-c', '--config', default='config-cifar.yaml',
                        type=str, help='Yaml config file')
    parser.add_argument('-g', '--gpu-id', default='2', type=str,
                        help='GPU id')
    parser.add_argument('-C', '--checkpoint', action='store_true',
                        help='Load from checkpoint stored in config["output_dir"]')

    return parser.parse_args()


def main(args):

    # ============================================================================
    # Config loading and CUDA settings
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    config['optim_parameters']['lr'] = eval(config['optim_parameters']['lr'])
    config['optim_parameters']['weight_decay'] = eval(config['optim_parameters']['weight_decay'])
    config['dataset']['input_shape'] = eval(config['dataset']['input_shape'])
    # import pdb; pdb.set_trace()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True

    if config['fp16_precision']:
        print('USING FP16 PRECISION')

    os.makedirs(config['output_dir'], exist_ok=True)
    with open('{}/config.yaml'.format(config['output_dir']), 'w') as file:
        yaml.dump(config, file)

    # ============================================================================
    # load dataset and step function
    dataloader = get_unsupervised_dataset(config['dataset'])
    simclr_step_function = simclr_step(config['loss']['temperature'], device)

    # ============================================================================
    # Network parameters
    model = SimCLR_Model(config['model'])
    n_parameters(sum(p.numel() for p in model.parameters()))

    # ============================================================================
    # Optimization Parameters
    optimizer = getattr(optim, config['optimizer'])
    optimizer = optimizer(model.parameters(), **config['optim_parameters'])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     T_max=(config['epochs'] -
                                                            config['warm_up']))

    # ============================================================================
    # Checkpoint
    if args.checkpoint:
        data = torch.load('{}/checkpoint.pth'.format(config['output_dir']),
                          map_location='cpu')
        model.load_state_dict(data['model'])

        optimizer.load_state_dict(data['optim'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

        init_epoch = data['epoch']
        csv = pd.read_csv('{}/summary.csv'.format(config['output_dir']))
        csv_eval = pd.read_csv('{}/evaluations.csv'.format(config['output_dir']))
    else:
        init_epoch = 0
        csv = pd.DataFrame(columns=['epoch', 'loss', 'lr'])
        csv_eval = pd.DataFrame(columns=['epoch', 'best epoch',
                                         'linear train acc',
                                         'linear test acc'])

    model.to(device)
    model.train()

    # ============================================================================
    # Optimization utils
    optim_utils = optimization()

    # ============================================================================
    # Learning loop
    epochs = config['epochs']
    for epoch in range(init_epoch, epochs):
        # warm up
        if epoch < config['warm_up']:
            for param_group in optimizer.param_groups:
                param_group['lr'] = (epoch + 1) * config['optim_parameters']['lr'] / config['warm_up']

        print('=' * 79)
        print(f'Epoch: {epoch + 1} / {epochs}')
        loss = optim_utils.train(model=model, device=device, optimizer=optimizer,
                                 step_class=simclr_step_function,
                                 train_loader=dataloader,
                                 use_mix_precision=config['fp16_precision'])
        csv = csv.append(pd.DataFrame({'epoch': [epoch + 1], 'loss': [loss], 'lr': [optimizer.param_groups[0]['lr'].item()]}),
                         ignore_index=True)
        csv.to_csv('{}/summary.csv'.format(config['output_dir']), index=False)

        # create checkpoint
        torch.save({'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                    'epoch': epoch}, '{}/checkpoint.pth'.format(config['output_dir']))

        if ((epoch + 1) % config['eval_every_n_epochs']) == 0:
            # evaluate
            top_epoch, best_train_acc, best_acc = linear_training(model, device,
                                                                  simclr_step_function,
                                                                  config, optim_utils)

            csv_eval = csv_eval.append(pd.DataFrame({'epoch': [epoch], 'best epoch': [top_epoch],
                                                     'linear train acc': [best_train_acc],
                                                     'linear test acc': [best_acc]}),
                                      ignore_index=True)

            csv_eval.to_csv('{}/evaluations.csv'.format(config['output_dir']), index=False)

        if epoch >= config['warm_up']:
            scheduler.step()

if __name__ == '__main__':
    args = arguments()
    main(args)