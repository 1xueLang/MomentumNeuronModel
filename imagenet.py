import os
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
import torchvision
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter

import snetx
from snetx import utils
import snetx.snn.algorithm as snnalgo

from snetx.dataset import vision as snnvds
from snetx.models import sew_resnet, ms_resnet, vgg
from snetx.snn import nwarmup
from snetx.training import classification as training
from snetx.snn import snnBN

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
    if neuron.startswith('b'):
        balancedMomentumLIF = __import__('balancedMomentumLIF')
        return balancedMomentumLIF.neuron.bMomentumLIF
    else:
        momentumLIF = __import__('momentumLIF')
        return momentumLIF.neuron.MomentumLIF

def execuate(device, args):
    tr_data, ts_data = snnvds.imagenet_dataset(args.data_dir, args.batch_size1, args.batch_size2)
    init_size = [1, args.T, 3, 224, 224]
    
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
    if 'sew' in args.arch:
        net = sew_resnet.__dict__[args.arch[3:]](import_neuron(args.neuron), neuron_cfg, norm_layer=snnBN.tdBN2d, num_classes=args.num_classes).to(device)
    elif 'ms' in args.arch:
        net = ms_resnet.__dict__[args.arch[2:]]('A', import_neuron(args.neuron), neuron_cfg, norm_layer=snnBN.tdBN2d, num_classes=args.num_classes).to(device)
    else:
        raise ValueError(f'{args.arch} is not supported.')
    
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
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate1, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD([{'params': pn, 'lr': args.learning_rate2, 'momentum': 0.9}, {'params': pw}], lr=args.learning_rate1, weight_decay=args.weight_decay, momentum=0.9)
        # optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate1, weight_decay=args.weight_decay, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max1, eta_min=args.eta_min)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)
    criterion = snnalgo.TET(torch.nn.CrossEntropyLoss(),)
    
    max_acc = 0.
    max_top5 = 0.
    for e in range(args.num_epochs):
        if args.verbose:
            dataloader = tqdm(tr_data)
        else:
            dataloader = tr_data
        
        correct, top5, sumup, loss = training.train_top5(net, dataloader, optimizer, criterion, scaler, device, args)
        correct, top5, sumup = training.validate_top5(net, tqdm(ts_data), device, args)
          
        if not args.debug:      
            writer.add_scalar('Loss', loss, e)
            writer.add_scalar('Acc', correct / sumup, e)
        
        if correct >= max_acc:
            torch.save(net.state_dict(), f'pths/imagenet_{args.arch}_M_{args.mo}_{args.learnable_mt}_l_{args.lamb}_{args.learnable_lb}_reset_{args.reset}_th_{args.th}.pth')

        max_acc = max(max_acc, correct)
        max_top5 = max(max_top5, top5)

        print('epoch: ', e, f'loss: {loss:.4f}, Acc: {(correct / sumup) * 100:.2f}%-{top5 / sumup * 100:.2f}%, Best: {(max_acc / sumup) * 100:.2f}%-{max_top5 / sumup * 100:.2f}%')  
        print(scheduler.get_last_lr(), neuron_cfg['alpha'].get_last_alpha())
        
        scheduler.step()
        neuron_cfg['alpha'].step()
    
    if not args.debug: writer.close()

def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size1', type=int, default=64, help='batch size for single device.')
    parser.add_argument('--batch_size2', type=int, default=128, help='test batch size.')
    parser.add_argument('--learning_rate1', type=float, default=1e-3, help='learning rate for gradient descent.')
    parser.add_argument('--learning_rate2', type=float, default=1e-3, help='learning rate for gradient descent.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='penal term parameter for model weight.')
    parser.add_argument('--optim', type=str, default='AdamW', help='AdamW, SGD')
    
    parser.add_argument('--neuron', type=str, default='bMomentumLIF')
    parser.add_argument('--tau', type=float, default=1., help='')
    parser.add_argument('--th', type=float, default=1., help='')
    parser.add_argument('--mo', type=float, default=0.5, help='')
    parser.add_argument('--lamb', type=float, default=0.5, help='')
    parser.add_argument('--learnable_mt', action='store_true', help='')
    parser.add_argument('--learnable_lb', action='store_true', help='')
    parser.add_argument('--reset', action='store_true')
    parser.add_argument('--regular', action='store_true')
    
    parser.add_argument('--T', type=int, default=6, help='snn simulate time step.')
    parser.add_argument('--num_epochs', type=int, default=200, help='max epochs for train process.')
    parser.add_argument('--T_max1', type=int, default=200, help='schedule period for consine annealing lr scheduler.')
    parser.add_argument('--T_max2', type=int, default=200, help='schedule period for consine annealing neuron warmup.')
    parser.add_argument('--eta_min', type=float, default=0.)
    parser.add_argument('--base', type=float, default=2., help='')
    parser.add_argument('--bound', type=float, default=2., help='')
    parser.add_argument('--drop', type=float, default=0.2, help='')

    parser.add_argument('--dataset', type=str, default='ImageNet', help='')
    parser.add_argument('--data_dir', type=str, default='../dataset', help='data directory.')
    parser.add_argument('--logs_dir', type=str, default='./LOGS/logs', help='logs directory.')
    parser.add_argument('--num_classes', type=int, default=1000)
    
    parser.add_argument('--print_intv', type=int, default=50, 
                        help='train steps interval to print train mesasges: show messages after each {intv} batches.')
    parser.add_argument('--amp', '-A', action='store_true')

    parser.add_argument('--verbose', '-V', action='store_true', 
                        help='whether to display training progress in the master process.')
    parser.add_argument('--arch', type=str, default='msresnet34', 
                        help='network architecture.'
                        )
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--debug', '-D', action='store_true')
    cmd_args = parser.parse_args()
    
    execuate(torch.device(cmd_args.device), cmd_args)
    

if __name__ == '__main__':
    main()
    # python imagenet.py -A -V