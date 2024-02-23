source ./scripts/base.sh

# runCIFAR10 $1 bMomentumLIF-cu 0.8 0. LOGS/spanLambda/ ms_resnet+resnet18 "--th 1.1 --batch_size1 64"
# runCIFAR10 $1 MomentumLIF-cu 0.2 0. LOGS/spanLambda/ ms_resnet+resnet18 "--th 1.1 --batch_size1 64"

# runCIFAR10 $1 MomentumLIF-cu 0.2 0. LOGS/spanLambda/ myVGG+vgg11_bn "--th 1.1 --batch_size1 64"


# runCIFAR10DVS $1 MomentumLIF-cu 0.2 0. LOGS/spanLambda/ "--th 1. --num_epochs 200 --T_max1 200 --T_max2 200 --learning_rate1 0.1 --batch_size2 128"
runCIFAR10DVS $1 bMomentumLIF-cu 0.8 0. LOGS/ "--th 1. --num_epochs 200 --T_max1 200 --T_max2 200 --learning_rate1 0.01 --learning_rate2 0.01 --batch_size2 128"