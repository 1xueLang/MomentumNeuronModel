import os
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
import torchvision
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter

import snetx
# from snetx import cuend
from snetx import utils as snutils
import snetx.snn.algorithm as snnalgo

from snetx.dataset import vision as snnvds
from snetx.snn import nwarmup
from snetx.training import classification as training
from snetx.snn import snnBN

from utils import myTransforms
import utils


def init_network(net, init_size, args):
    net.eval()
    with torch.no_grad():
        net(torch.rand(init_size).to(args.device))
    
    print(net)
    net.train()


def import_neuron(neuron):
    if not neuron.endswith('cu'):
        return utils.neuron.__dict__[neuron]
    
    if 'Parametric' in neuron:
        parametricLIF = __import__('parametricLIF')
        return parametricLIF.neuron.__dict__[neuron[:-3]]

    if neuron.startswith('b'):
        balancedMomentumLIF = __import__('balancedMomentumLIF')
        return balancedMomentumLIF.neuron.bMomentumLIF
    else:
        momentumLIF = __import__('momentumLIF')
        return momentumLIF.neuron.MomentumLIF


def execuate(device, args):
    if args.seed > 0:
        snutils.seed_all(args.seed)
        cuend = __import__('snetx.cuend')
        cuend.utils.seed_cupy(args.seed)

    if args.dataset == 'CIFAR10':
        tr_data, ts_data = snnvds.cifar10_dataset(args.data_dir, args.batch_size1, args.batch_size2, myTransforms.cifar10_transforms(True, True))
        init_size = [1, args.T, 3, 32, 32]
        dvs = False
    elif args.dataset == 'CIFAR100':
        tr_data, ts_data = snnvds.cifar100_dataset(args.data_dir, args.batch_size1, args.batch_size2)
        init_size = [1, args.T, 3, 32, 32]
        dvs = False
    elif args.dataset == 'CIFAR10DVS':
        tr_data, ts_data = snnvds.cifar10dvs_dataset(args.data_dir, args.batch_size1, args.batch_size2, args.T)
        init_size = [1, args.T, 2, 48, 48]
        dvs = True
    elif args.dataset == 'DVSGesture128':
        tr_data, ts_data = snnvds.dvs128gesture_dataset(args.data_dir, args.batch_size1, args.batch_size2, args.T)
        init_size = [1, args.T, 2, 128, 128]
        dvs = True
    else:
        raise ValueError(f'{args.dataset} not supported.')
    
    neuron_cfg = {
        'alpha': nwarmup.PolynormialWarmup(args.base, args.bound, args.T_max2),
        'tau': args.tau,
        'th': args.th,
        'momentum': args.mo,
        'lamb': args.lamb,
        'learnable_mt': args.learnable_mt,
        'learnable_lb': args.learnable_lb,
        'regular': args.regular,
    }
    
    net = utils.arch_parser.get_network(args, import_neuron(args.neuron), neuron_cfg, snnBN.tdBN2d).to(device)
    init_network(net, init_size, args)
    
    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None
    
    if args.debug:
        writer = None
    else:
        time_str = datetime.now().strftime('%Y%m%d%H%M%S')
        writer = SummaryWriter(log_dir=f'{args.logs_dir}/{args.dataset}/{args.arch}/tau-{args.tau}-th-{args.th}/{args.neuron}/L-{args.lamb}{args.learnable_lb}/M-{args.mo}{args.learnable_mt}/r-{args.regular}/{time_str}')
    
    pw = []
    pn = []
    for n, p in net.named_parameters():
        if 'lamb' in n or 'momentum' in n:
            pn.append(p)
        else:
            pw.append(p)
    
    if args.optim == 'AdamW':
        optimizer = torch.optim.AdamW(
            [{'params': pn, 'lr': args.learning_rate2, 'momentum': 0.9}, {'params': pw}], 
            lr=args.learning_rate1, weight_decay=args.weight_decay,
        )
    else:
        # optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)
        optimizer = torch.optim.SGD(
            [{'params': pn, 'lr': args.learning_rate2, 'momentum': 0.9}, {'params': pw}], 
            lr=args.learning_rate1, weight_decay=args.weight_decay, momentum=0.9
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max1, eta_min=args.eta_min)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)
    criterion = snnalgo.TET(torch.nn.CrossEntropyLoss(),)
    
    max_acc = 0.
    for e in range(args.num_epochs):
        if args.verbose:
            dataloader = tqdm(tr_data)
        else:
            dataloader = tr_data
        
        if dvs:
            correct, sumup, loss = training.train_dvs(net, dataloader, optimizer, criterion, scaler, device, args)
            correct, sumup = training.validate(net, ts_data, device, args, static=False)
        else:
            correct, sumup, loss = training.train_static(net, dataloader, optimizer, criterion, scaler, device, args)
            correct, sumup = training.validate(net, tqdm(ts_data), device, args)
          
        if not args.debug:      
            writer.add_scalar('Loss', loss, e)
            writer.add_scalar('Acc', correct / sumup, e)
        
        if args.save and max_acc <= correct:
            torch.save(net.state_dict(), f'pths/{args.dataset}_{args.arch}_{args.neuron}_M{args.mo:.2f}{args.learnable_mt}_L{args.lamb:.2f}{args.learnable_lb}.pth')
            
        max_acc = max(max_acc, correct)

        print('epoch: ', e, f'loss: {loss:.4f}, Acc: {(correct / sumup) * 100:.2f}%, Best: {(max_acc / sumup) * 100:.2f}%')  
        print(scheduler.get_last_lr(), neuron_cfg['alpha'].get_last_alpha())
        
        scheduler.step()
        neuron_cfg['alpha'].step()
    
    if not args.debug:
        writer.close()

def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size1', type=int, default=128, help='batch size for single device.')
    parser.add_argument('--batch_size2', type=int, default=256, help='test batch size.')
    parser.add_argument('--learning_rate1', type=float, default=0.01, help='learning rate for gradient descent.')
    parser.add_argument('--learning_rate2', type=float, default=0.01, help='learning rate for gradient descent.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='penal term parameter for model weight.')
    parser.add_argument('--optim', type=str, default='SGD', help='AdamW, SGD')
    parser.add_argument('--num_epochs', type=int, default=400, help='max epochs for train process.')
    parser.add_argument('--T_max1', type=int, default=400, help='schedule period for consine annealing lr scheduler.')
    parser.add_argument('--T_max2', type=int, default=400, help='schedule period for consine annealing neuron warmup.')
    parser.add_argument('--eta_min', type=float, default=0.)
    parser.add_argument('--base', type=float, default=1., help='')
    parser.add_argument('--bound', type=float, default=1., help='')
    parser.add_argument('--print_intv', type=int, default=50, 
                        help='train steps interval to print train mesasges: show messages after each {intv} batches.')
    
    parser.add_argument('--neuron', type=str, default='bMomentumLIF-cu')
    parser.add_argument('--tau', type=float, default=2., help='')
    parser.add_argument('--th', type=float, default=1., help='')
    parser.add_argument('--mo', type=float, default=0., help='')
    parser.add_argument('--lamb', type=float, default=.8, help='')
    parser.add_argument('--learnable_mt', action='store_true', help='')
    parser.add_argument('--learnable_lb', action='store_true', help='')
    parser.add_argument('--regular', action='store_true')
    
    parser.add_argument('--T', type=int, default=6, help='snn simulate time step.')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='')
    parser.add_argument('--data_dir', type=str, default='../dataset', help='data directory.')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--logs_dir', type=str, default='./LOGS/logs', help='logs directory.')
    
    parser.add_argument('--arch', type=str, default='ms_resnet+resnet18', help='network architecture.')
    parser.add_argument('--drop', type=float, default=0.2, help='')

    parser.add_argument('--seed', type=int, default=-1, help='')
    parser.add_argument('--amp', '-A', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--debug', '-D', action='store_true')
    parser.add_argument('--verbose', '-V', action='store_true', 
                        help='whether to display training progress in the master process.')
    parser.add_argument('--save', '-S', action='store_true',)

    cmd_args = parser.parse_args()
    
    execuate(torch.device(cmd_args.device), cmd_args)
    

if __name__ == '__main__':
    main()
