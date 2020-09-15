import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# mix precision imports
from torch.cuda.amp import GradScaler, autocast

# custom imports
from core.misc import Metric_Logger
from core.datasets import get_supervised_dataset

class optimization():
    def __init__(self):
        self.logger = Metric_Logger()
        self.lt_logger = Metric_Logger()
        self.vt_logger = Metric_Logger()

    def train(self, model, device, optimizer, step_class,
              train_loader, use_mix_precision=False):

        self.logger.restart(len(train_loader))

        if not model.training:
            model.train()

        if use_mix_precision:
            scaler = GradScaler()

        for idx, (im1, im2) in enumerate(train_loader):

            im1 = im1.to(device)
            im2 = im2.to(device)

            if use_mix_precision:
                with autocast():
                    im1 = im1.to(device, dtype=torch.float)
                    im2 = im2.to(device, dtype=torch.float)
                    loss = step_class(phase='unsupervised',
                                      im1=im1, im2=im2, model=model)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                im1 = im1.to(device, dtype=torch.float)
                im2 = im2.to(device, dtype=torch.float)
                loss = step_class(phase='unsupervised',
                                  im1=im1, im2=im2, model=model)
                loss.backward()
                optimizer.step()
            
            optimizer.zero_grad()

            self.logger.add_metric(loss.item(), im1.size(0) * 2)
            self.logger.print_progress(idx)

        self.logger.print_progress(idx, True)

        return self.logger.get_mean()

    def train_linear(self, model, linear, device, optimizer, step_class,
                     train_loader, epoch=None, epochs=None):

        linear.train()

        self.lt_logger.restart(len(train_loader))

        for idx, (img, lbl) in enumerate(train_loader):

            img = img.to(device, dtype=torch.float)
            lbl = lbl.to(device, dtype=torch.long)


            loss, top1 = step_class(phase='linear',
                                    img=img, lbl=lbl,
                                    model=model, linear=linear)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            self.lt_logger.add_metric_and_top1(loss.item(), top1,
                                               img.size(0))
            self.lt_logger.print_linear_progress(idx, epoch, epochs)

        self.lt_logger.print_linear_progress(idx, epoch, epochs, True)

        return self.lt_logger.get_mean(), self.lt_logger.get_acc()

    def eval_linear(self, model, linear, device, step_class,
                    val_loader, epoch=None, epochs=None):

        linear.eval()

        self.vt_logger.restart(len(val_loader))

        for idx, (img, lbl) in enumerate(val_loader):

            img = img.to(device, dtype=torch.float)
            lbl = lbl.to(device, dtype=torch.long)

            loss, top1 = step_class(phase='linear test',
                                    img=img, lbl=lbl,
                                    model=model, linear=linear)

            self.vt_logger.add_metric_and_top1(loss.item(), top1,
                                               img.size(0))
            self.vt_logger.print_linear_progress(idx, epoch, epochs)

        self.vt_logger.print_linear_progress(idx, epoch, epochs, True)

        return self.vt_logger.get_mean(), self.vt_logger.get_acc()


############################
### EVALUATION FUNCTIONS ###
############################


def linear_training(model, device, step_class, config, optim_utils):

    print('-' * 79)
    print('Performing linear evaluation')

    model.eval()

    linear_classifier = torch.nn.Linear(model.backbone.output_dim,
                                        config['dataset']['num_classes'],
                                        bias=True)
    linear_classifier.to(device)

    train_loader, val_loader = get_supervised_dataset(config['dataset'])

    epochs = config['epochs_linear']

    optimizer = getattr(optim, config['optimizer_eval'])
    optimizer = optimizer(linear_classifier.parameters(), **config['optim_eval_param'])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     T_max=epochs)

    criterion = torch.nn.CrossEntropyLoss()
    best_acc = 0
    best_train_acc = 0
    top_epoch = -1

    for epoch in range(epochs):

        tl, tt1 = optim_utils.train_linear(model, linear_classifier,
                                           device, optimizer,
                                           step_class, train_loader,
                                           epoch, epochs)

        vl, vt1 = optim_utils.eval_linear(model, linear_classifier,
                                          device, step_class,
                                          val_loader, epoch, epochs)

        if best_acc < vt1:
            best_acc = vt1
            best_train_acc = tt1
            top_epoch = epoch

        scheduler.step()

    return top_epoch, best_train_acc, best_acc
