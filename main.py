import torch
import torch.nn as nn
import os
import time
import matplotlib.pyplot as plt

from data.loaders import get_cifar_loader
from models.CIFARResNet import CIFARResNet, TinyResNet
from models.vgg import VGG_A, VGG_A_Light, VGG_A_Dropout, VGG_A_BatchNorm
from utils.trainer import Trainer, Validator
from utils.recorder import plot_loss, plot_acc, plot_comparison, record_logs, save_model


class runner():

    def __init__(self):
        pass


    def run(self, model_n, op_n, params):
        # params = [bs, lr, epochs, weight_decay, mixup_alpha]
        bs, lr, epochs, weight_decay, mixup_alpha = params
        warmup_epochs = 5

        start_time = time.time()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device:{device}')
        
        train_loader = get_cifar_loader(batch_size=bs,train=True)
        test_loader = get_cifar_loader(batch_size=bs,train=False)

        if model_n == 'resnet':
            model = CIFARResNet()
        elif model_n == 'tinyresnet':
            model = TinyResNet()
        elif model_n == 'vgg_a':
            model = VGG_A()
        elif model_n == 'vgg_a_bn':
            model = VGG_A_BatchNorm()
        else:
            raise ValueError("Model type not supported.")
        
        if op_n == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
                nesterov=True,
            )
        elif op_n == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError("Optimizer type not supported.")
        
        if warmup_epochs > 0:
            base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, epochs - warmup_epochs)
            )
            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: float(epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0
            )
            # Use chainable scheduler form and call scheduler.step() without passing epoch.
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, base_scheduler],
                milestones=[warmup_epochs]
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs
            )

        criterion = nn.CrossEntropyLoss(
            label_smoothing=0.1
        )

        trainer = Trainer(
            model, train_loader, test_loader, criterion,
            optimizer, scheduler, device, epochs=epochs, 
            print_freq=100, mixup_alpha=mixup_alpha
            )
        
        trainer.train()

        valid = Validator(model, test_loader, device)
        valid.accuracy()

        end_time = time.time()

        plot_loss(trainer.losses, valid.acc, types=[model_n, op_n, mixup_alpha])
        plot_acc(trainer.acc, valid.acc, types=[model_n, op_n, mixup_alpha])
        record_logs(params, valid.acc, types=[model_n, op_n, mixup_alpha], time=end_time-start_time)
        save_model(model, types=[model_n, op_n, mixup_alpha])

        print(f'Final train accuracy: {trainer.acc[-1]:.4f}')
        print(f'Final test accuracy: {valid.acc:.4f}')
        return valid.acc, trainer.losses, trainer.acc



if __name__ == "__main__":
    
    bs = 256
    lr = 0.01
    epochs = 100
    weight_decay = 5e-4
    mixup_alpha = 1.0
    params1 = [bs, lr*2, epochs, weight_decay, mixup_alpha]
    params2 = [bs, lr, epochs, weight_decay, mixup_alpha]

    r = runner()

    _, loss1, acc1 = r.run(model_n='resnet', op_n='sgd', params=params1)
    _, loss2, acc2 = r.run(model_n='resnet', op_n='adamw', params=params2)
    _, loss3, acc3 = r.run(model_n='tinyresnet', op_n='sgd', params=params1)
    _, loss4, acc4 = r.run(model_n='tinyresnet', op_n='adamw', params=params2)

    plot_comparison(
        [
            (loss1, 'resnet-sgd'),
            (loss2, 'resnet-adamw'),
            (loss3, 'tinyresnet-sgd'),
            (loss4, 'tinyresnet-adamw')

        ],
        title='Loss Comparison',
        ylabel='Loss'
    )

    plot_comparison(
        [
            (acc1, 'resnet-sgd'),
            (acc2, 'resnet-adamw'),
            (acc3, 'tinyresnet-sgd'),
            (acc4, 'tinyresnet-adamw')
        ],
        title='Accuracy Comparison',
        ylabel='Train Accuracy'
    )

    
    # r.run(params= params, model_n='vgg_a', op_n='adamw', params=params)
    # r.run(params=params,model_n='vgg_a_bn', op_n='adamw', params=params)

