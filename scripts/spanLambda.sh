source ./scripts/base.sh

# 实验1：测试λ取不同值时两个神经元在静态和神经形态数据集上的表现，并分别保存模型后续测试点火率情况

function spanLambda() {
    for lamb in `seq $3 $4 $5`; do
        runCIFAR10 $1 $2 $lamb 0. LOGS/spanLambda/ ms_resnet+resnet18 "-S --th 1.1 --batch_size1 64"
    done
}

function spanLambdaDvs() {
    for lamb in `seq $3 $4 $5`; do
        runCIFAR10DVS $1 $2 $lamb 0. LOGS/spanLambdaDvs/ "-S --th 1. --num_epochs 200 --T_max1 200 --T_max2 200 --learning_rate1 0.05 --batch_size2 128"
    done
}

# ./scripts/spanLambda.sh spanLambda 0 MomentumLIF-cu 0.0 0.05 0.4
# ./scripts/spanLambda.sh spanLambdaDvs 0 bMomentumLIF-cu 0.75 0.05 0.95

eval $1 $2 $3 $4 $5 $6
