runCIFAR10() {
    CUDA_VISIBLE_DEVICES=$1                         \
    python main.py --neuron $2 --lamb $3 --mo $4    \
                   --logs_dir $5 --arch $6          \
                   --tau 2. -V $7
}

runCIFAR10DVS() {
    CUDA_VISIBLE_DEVICES=$1                         \
    python main.py --neuron $2 --lamb $3 --mo $4    \
                   --logs_dir $5                    \
                   --arch myVGG+vgg11_bn --drop 0.3 \
                   --T 10 --dataset CIFAR10DVS      \
                   --tau 4. --batch_size1 32 -V $6
}