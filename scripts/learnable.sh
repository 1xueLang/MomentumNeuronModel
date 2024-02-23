source ./scripts/base.sh

# 实验2：可学习参数的影响，分静态和神经形态数据集具有不同的测试重点,保存模型
#       静态数据集：需要更长的训练周期，因此epochs增加
#       神经形态数据集：收敛速度快，因此测试最终准确率，不需要加长训练周期

runCIFAR10 $1 $2 $3 0. LOGS/learnable/ ms_resnet+resnet18 "-S --th 1.1 --batch_size1 64 --num_epochs 1024 --T_max1 1024 --T_max2 1024 "$4

# LIF
# ./scripts/learnable.sh 0 MomentumLIF-cu 0. ""
# PLIF
# ./scripts/learnable.sh 0 ParametricLIF-cu 0. ""

# bMLIF
# ./scripts/learnable.sh 0 bMomentumLIF-cu 0.85 "--learnable_lb"
# bmPLIF unlearnable
# ./scripts/learnable.sh 0 bmParametricLIF-cu 0.85 "" 0
# bmPLIF 
# ./scripts/learnable.sh 0 bmParametricLIF-cu 0.85 "--learnable_lb" 1