import os
import unittest

import torch
import torch.nn as nn
from snetx.cuend.neuron import LIF

import utils
import balancedMomentumLIF
import momentumLIF
import parametricLIF
from colorama import Fore, init


device = 1
size = [32, 10, 32 * 32 * 32]

def print_distance(v1, v2, desc):
    print(f'Sum      [{desc}]:', v1.sum().item(), v2.sum().item())
    print(f'Distance [{desc}]:', torch.pairwise_distance(v1.flatten().view(1, -1), v2.flatten().view(1, -1)).item())
    print(Fore.RED, f'Cosine  [{desc}]:', torch.cosine_similarity(v1.flatten().view(1, -1), v2.flatten().view(1, -1)).item(), Fore.RESET)
    print(f'==')


class MomentumLIFTestCase(unittest.TestCase):
    def setUp(self):
        self.x1 = torch.rand(size).to(device) * 2
        self.x2 = self.x1.clone().detach()
        self.x1.requires_grad = True
        self.x2.requires_grad = True
    
    def case0(self, neuron_kwargs):
        tn = utils.neuron.MomentumLIF(**neuron_kwargs).to(device)
        cn = momentumLIF.neuron.MomentumLIF(**neuron_kwargs).to(device)
        self.x1.grad = None
        self.x2.grad = None
        s1 = tn(self.x1)
        s2 = cn(self.x2)
        s1.sum().backward()
        s2.sum().backward()
        print()
        print_distance(self.x1, self.x2, ' x')
        print_distance(s1, s2, ' s')
        print_distance(self.x1.grad, self.x2.grad, 'dx')
        print_distance(tn.lamb.grad, cn.lamb.grad, 'dl')
        print_distance(tn.momentum.grad, cn.momentum.grad, 'dm')

    def testCase0(self):
        neuron_kwargs = dict(learnable_lb=True, lamb=0.4, learnable_mt=True, momentum=0.4)
        self.case0(neuron_kwargs)

    def testCase0Regular(self):
        neuron_kwargs = dict(learnable_lb=True, lamb=0.4, learnable_mt=True, momentum=0.4, regular=True)
        self.case0(neuron_kwargs)

    def case1(self, neuron_kwargs):
        tn = utils.neuron.MomentumLIF(**neuron_kwargs).to(device)
        cn = momentumLIF.neuron.MomentumLIF(**neuron_kwargs).to(device)
        self.x1.grad = None
        self.x2.grad = None
        s1 = tn(self.x1)
        s2 = cn(self.x2)
        s1.sum().backward()
        s2.sum().backward()
        print()
        print_distance(self.x1, self.x2, ' x')
        print_distance(s1, s2, ' s')
        print_distance(self.x1.grad, self.x2.grad, 'dx')
        print_distance(tn.lamb.grad, cn.lamb.grad, 'dl')

    def testCase1(self):
        neuron_kwargs = dict(learnable_lb=True, lamb=0.4, learnable_mt=False, momentum=0.4)
        self.case1(neuron_kwargs)

    def testCase1Regular(self):
        neuron_kwargs = dict(learnable_lb=True, lamb=0.4, learnable_mt=False, momentum=0.4, regular=True)
        self.case1(neuron_kwargs)

    def case2(self, neuron_kwargs):
        tn = utils.neuron.MomentumLIF(**neuron_kwargs).to(device)
        cn = momentumLIF.neuron.MomentumLIF(**neuron_kwargs).to(device)
        self.x1.grad = None
        self.x2.grad = None
        s1 = tn(self.x1)
        s2 = cn(self.x2)
        s1.sum().backward()
        s2.sum().backward()
        print()
        print_distance(self.x1, self.x2, ' x')
        print_distance(s1, s2, ' s')
        print_distance(self.x1.grad, self.x2.grad, 'dx')
        print_distance(tn.momentum.grad, cn.momentum.grad, 'dm')

    def testCase2(self):
        neuron_kwargs = dict(learnable_lb=False, lamb=0.4, learnable_mt=True, momentum=0.4)
        self.case2(neuron_kwargs)

    def testCase2Regular(self):
        neuron_kwargs = dict(learnable_lb=False, lamb=0.4, learnable_mt=True, momentum=0.4, regular=True)
        self.case2(neuron_kwargs) 

    def case3(self, neuron_kwargs):
        tn = utils.neuron.MomentumLIF(**neuron_kwargs).to(device)
        cn = momentumLIF.neuron.MomentumLIF(**neuron_kwargs).to(device)
        self.x1.grad = None
        self.x2.grad = None
        s1 = tn(self.x1)
        s2 = cn(self.x2)
        s1.sum().backward()
        s2.sum().backward()
        print()
        print_distance(self.x1, self.x2, ' x')
        print_distance(s1, s2, ' s')
        print_distance(self.x1.grad, self.x2.grad, 'dx')

    def testCase3(self):
        neuron_kwargs = dict(learnable_lb=False, lamb=0.4, learnable_mt=False, momentum=0.4)
        self.case3(neuron_kwargs)

    def testCase3Regular(self):
        neuron_kwargs = dict(learnable_lb=False, lamb=0.4, learnable_mt=False, momentum=0.4, regular=True)
        self.case3(neuron_kwargs)

    def testLIF(self):
        neuron_kwargs = dict(learnable_lb=False, lamb=0., learnable_mt=False, momentum=0.4, regular=True)
        self.case3(neuron_kwargs)


class bMomentumLIFTestCase(unittest.TestCase):
    def setUp(self):
        self.x1 = torch.rand(size).to(device) * 2
        self.x2 = self.x1.clone().detach()
        self.x1.requires_grad = True
        self.x2.requires_grad = True
    
    def case0(self, neuron_kwargs):
        tn = utils.neuron.bMomentumLIF(**neuron_kwargs).to(device)
        cn = balancedMomentumLIF.neuron.bMomentumLIF(**neuron_kwargs).to(device)
        self.x1.grad = None
        self.x2.grad = None
        s1 = tn(self.x1)
        s2 = cn(self.x2)
        s1.sum().backward()
        s2.sum().backward()
        print()
        print_distance(self.x1, self.x2, ' x')
        print_distance(s1, s2, ' s')
        print_distance(self.x1.grad, self.x2.grad, 'dx')
        print_distance(tn.lamb.grad, cn.lamb.grad, 'dl')
        print_distance(tn.momentum.grad, cn.momentum.grad, 'dm')

    def testCase0(self):
        neuron_kwargs = dict(learnable_lb=True, lamb=0.4, learnable_mt=True, momentum=0.4)
        self.case0(neuron_kwargs)

    def testCase0Regular(self):
        neuron_kwargs = dict(learnable_lb=True, lamb=0.4, learnable_mt=True, momentum=0.4, regular=True)
        self.case0(neuron_kwargs)

    def case1(self, neuron_kwargs):
        tn = utils.neuron.bMomentumLIF(**neuron_kwargs).to(device)
        cn = balancedMomentumLIF.neuron.bMomentumLIF(**neuron_kwargs).to(device)
        self.x1.grad = None
        self.x2.grad = None
        s1 = tn(self.x1)
        s2 = cn(self.x2)
        s1.sum().backward()
        s2.sum().backward()
        print()
        print_distance(self.x1, self.x2, ' x')
        print_distance(s1, s2, ' s')
        print_distance(self.x1.grad, self.x2.grad, 'dx')
        print_distance(tn.lamb.grad, cn.lamb.grad, 'dl')

    def testCase1(self):
        neuron_kwargs = dict(learnable_lb=True, lamb=0.4, learnable_mt=False, momentum=0.4)
        self.case1(neuron_kwargs)

    def testCase1Regular(self):
        neuron_kwargs = dict(learnable_lb=True, lamb=0.4, learnable_mt=False, momentum=0.4, regular=True)
        self.case1(neuron_kwargs)

    def case2(self, neuron_kwargs):
        tn = utils.neuron.bMomentumLIF(**neuron_kwargs).to(device)
        cn = balancedMomentumLIF.neuron.bMomentumLIF(**neuron_kwargs).to(device)
        self.x1.grad = None
        self.x2.grad = None
        s1 = tn(self.x1)
        s2 = cn(self.x2)
        s1.sum().backward()
        s2.sum().backward()
        print()
        print_distance(self.x1, self.x2, ' x')
        print_distance(s1, s2, ' s')
        print_distance(self.x1.grad, self.x2.grad, 'dx')
        print_distance(tn.momentum.grad, cn.momentum.grad, 'dm')

    def testCase2(self):
        neuron_kwargs = dict(learnable_lb=False, lamb=0.4, learnable_mt=True, momentum=0.4)
        self.case2(neuron_kwargs)

    def testCase2Regular(self):
        neuron_kwargs = dict(learnable_lb=False, lamb=0.4, learnable_mt=True, momentum=0.4, regular=True)
        self.case2(neuron_kwargs) 

    def case3(self, neuron_kwargs):
        tn = utils.neuron.bMomentumLIF(**neuron_kwargs).to(device)
        cn = balancedMomentumLIF.neuron.bMomentumLIF(**neuron_kwargs).to(device)
        self.x1.grad = None
        self.x2.grad = None
        s1 = tn(self.x1)
        s2 = cn(self.x2)
        s1.sum().backward()
        s2.sum().backward()
        print()
        print_distance(self.x1, self.x2, ' x')
        print_distance(s1, s2, ' s')
        print_distance(self.x1.grad, self.x2.grad, 'dx')

    def testCase3(self):
        neuron_kwargs = dict(learnable_lb=False, lamb=0.4, learnable_mt=False, momentum=0.4)
        self.case3(neuron_kwargs)

    def testCase3Regular(self):
        neuron_kwargs = dict(learnable_lb=False, lamb=0.4, learnable_mt=False, momentum=0.4, regular=True)
        self.case3(neuron_kwargs)

    def testLIF(self):
        neuron_kwargs = dict(learnable_lb=False, lamb=1., learnable_mt=False, momentum=0.4, regular=True)
        self.case3(neuron_kwargs)


class ParametricLIFTestCase(unittest.TestCase):
    def setUp(self):
        self.x1 = torch.rand(size).to(device) * 2
        self.x2 = self.x1.clone().detach()
        self.x1.requires_grad = True
        self.x2.requires_grad = True

    def testParametricLIF(self):
        tn = utils.neuron.ParametricLIF().to(device)
        cn = parametricLIF.neuron.ParametricLIF().to(device)
        self.x1.grad = None
        self.x2.grad = None
        s1 = tn(self.x1)
        s2 = cn(self.x2)
        s1.sum().backward()
        s2.sum().backward()
        print()
        print_distance(self.x1, self.x2, ' x')
        print_distance(s1, s2, ' s')
        print_distance(self.x1.grad, self.x2.grad, 'dx')
        print_distance(tn.tau.grad, cn.tau.grad, 'dtau')

    def testmParametricLIF(self):
        tn = utils.neuron.mParametricLIF().to(device)
        cn = parametricLIF.neuron.mParametricLIF().to(device)
        self.x1.grad = None
        self.x2.grad = None
        s1 = tn(self.x1)
        s2 = cn(self.x2)
        s1.sum().backward()
        s2.sum().backward()
        print()
        print_distance(self.x1, self.x2, ' x')
        print_distance(s1, s2, ' s')
        print_distance(self.x1.grad, self.x2.grad, 'dx')
        print_distance(tn.tau.grad, cn.tau.grad, 'dtau')
        print_distance(tn.lamb.grad, cn.lamb.grad, 'dlamb')

    def testbmParametricLIF(self):
        tn = utils.neuron.bmParametricLIF().to(device)
        cn = parametricLIF.neuron.bmParametricLIF().to(device)
        self.x1.grad = None
        self.x2.grad = None
        s1 = tn(self.x1)
        s2 = cn(self.x2)
        s1.sum().backward()
        s2.sum().backward()
        print()
        print_distance(self.x1, self.x2, ' x')
        print_distance(s1, s2, ' s')
        print_distance(self.x1.grad, self.x2.grad, 'dx')
        print_distance(tn.tau.grad, cn.tau.grad, 'dtau')
        print_distance(tn.lamb.grad, cn.lamb.grad, 'dlamb')

if __name__ == '__main__':
    os.system('clear')
    init()
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    # suite.addTest(loader.loadTestsFromTestCase(MomentumLIFTestCase))
    # suite.addTest(loader.loadTestsFromTestCase(bMomentumLIFTestCase))
    suite.addTest(loader.loadTestsFromTestCase(ParametricLIFTestCase))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
