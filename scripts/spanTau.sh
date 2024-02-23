source ./scripts/base.sh

# 实验3：对tau的鲁棒性：分别测试λ学习与不学习，两个数据集，两个神经元，tau取不同值
function spanTau() {
    for lamb in 1 2 4 8; do
        runCIFAR10 $1 $2 $3 0. LOGS/spanTau/ ms_resnet+resnet18 "-S --th 1.1 --batch_size1 64"
        runCIFAR10 $1 $2 $4 0. LOGS/spanTau/ ms_resnet+resnet18 "-S --th 1.1 --batch_size1 64"
    done
}

function spanTauDvs() {
    for lamb in 1 2 4 8; do
        runCIFAR10DVS $1 $2 $3 0. LOGS/spanTauDvs/ "-S --th 1. --num_epochs 150 --T_max1 150 --T_max2 150 --learning_rate1 0.05 --batch_size2 128"
    done
}

# ./scripts/spanLambda.sh spanTau 0 MomentumLIF-cu 0.2 0.0
# ./scripts/spanLambda.sh spanTauDvs 0 bMomentumLIF-cu 0.8 1.0

eval $1 $2 $3 $4