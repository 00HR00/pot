#!/bin/bash

echo "Starting the script"

python buffer_honest.py --dataset=CIFAR10 --model=ConvNetD2 --zca
python buffer_honest.py --dataset=CIFAR10 --model=ResNet18 --zca
python buffer_honest.py --dataset=CIFAR10 --model=ResNet34 --zca
python buffer_honest.py --dataset=CIFAR100 --model=ConvNetD2 --zca
python buffer_honest.py --dataset=CIFAR100 --model=ResNet18 --zca
python buffer_honest.py --dataset=CIFAR100 --model=ResNet34 --zca

python buffer_attacker_1.py --dataset=CIFAR100 --model=ResNet34
python buffer_attacker_1.py --dataset=CIFAR10 --model=ConvNetD2
python buffer_attacker_1.py --dataset=CIFAR100 --model=ConvNetD2
python buffer_attacker_1.py --dataset=CIFAR10 --model=ResNet34
python buffer_attacker_1.py --dataset=CIFAR100 --model=ResNet18
python buffer_attacker_1.py --dataset=CIFAR10 --model=ResNet18

python buffer_attacker_2.py --dataset=CIFAR100 --model=ResNet34
python buffer_attacker_2.py --dataset=CIFAR10 --model=ConvNetD2
python buffer_attacker_2.py --dataset=CIFAR100 --model=ConvNetD2
python buffer_attacker_2.py --dataset=CIFAR10 --model=ResNet34
python buffer_attacker_2.py --dataset=CIFAR100 --model=ResNet18
python buffer_attacker_2.py --dataset=CIFAR10 --model=ResNet18

python buffer_attacker_2_long.py --dataset=CIFAR100 --model=ResNet34
python buffer_attacker_2_long.py --dataset=CIFAR10 --model=ConvNetD2
python buffer_attacker_2_long.py --dataset=CIFAR100 --model=ConvNetD2
python buffer_attacker_2_long.py --dataset=CIFAR10 --model=ResNet34
python buffer_attacker_2_long.py --dataset=CIFAR100 --model=ResNet18
python buffer_attacker_2_long.py --dataset=CIFAR10 --model=ResNet18

python buffer_attacker_5.py --dataset=CIFAR100 --model=ResNet34 --train_epochs=50 --num_experts=3 --zca
python buffer_attacker_5.py --dataset=CIFAR10 --model=ConvNetD2 --train_epochs=50 --num_experts=3 --zca
python buffer_attacker_5.py --dataset=CIFAR100 --model=ConvNetD2 --train_epochs=50 --num_experts=3 --zca
python buffer_attacker_5.py --dataset=CIFAR10 --model=ResNet34 --train_epochs=50 --num_experts=3 --zca
python buffer_attacker_5.py --dataset=CIFAR100 --model=ResNet18 --train_epochs=50 --num_experts=3 --zca
python buffer_attacker_5.py --dataset=CIFAR10 --model=ResNet18 --train_epochs=50 --num_experts=3 --zca

python buffer_attacker_5_short.py --dataset=CIFAR100 --model=ResNet34 --train_epochs=50 --num_experts=2 --zca
python buffer_attacker_5_short.py --dataset=CIFAR10 --model=ConvNetD2 --train_epochs=50 --num_experts=2 --zca
python buffer_attacker_5_short.py --dataset=CIFAR100 --model=ConvNetD2 --train_epochs=50 --num_experts=2 --zca
python buffer_attacker_5_short.py --dataset=CIFAR10 --model=ResNet34 --train_epochs=50 --num_experts=2 --zca
python buffer_attacker_5_short.py --dataset=CIFAR100 --model=ResNet18 --train_epochs=50 --num_experts=2 --zca
python buffer_attacker_5_short.py --dataset=CIFAR10 --model=ResNet18 --train_epochs=50 --num_experts=2 --zca

python buffer_attacker_6.py --dataset=CIFAR10 --model=ResNet18 --train_epochs=50 --num_experts=2 --zca
python buffer_attacker_6.py --dataset=CIFAR10 --model=ResNet34 --train_epochs=50 --num_experts=2 --zca
python buffer_attacker_6.py --dataset=CIFAR10 --model=ConvNetD2 --train_epochs=50 --num_experts=2 --zca
python buffer_attacker_6.py --dataset=CIFAR100 --model=ResNet18 --train_epochs=50 --num_experts=2 --zca
python buffer_attacker_6.py --dataset=CIFAR100 --model=ResNet34 --train_epochs=50 --num_experts=2 --zca
python buffer_attacker_6.py --dataset=CIFAR100 --model=ConvNetD2 --train_epochs=50 --num_experts=2 --zca

python buffer_attacker_6_short_50.py --dataset=CIFAR10 --model=ResNet18 --train_epochs=50 --num_experts=1 --zca
python buffer_attacker_6_short_50.py --dataset=CIFAR10 --model=ResNet34 --train_epochs=50 --num_experts=1 --zca
python buffer_attacker_6_short_50.py --dataset=CIFAR100 --model=ConvNetD2 --train_epochs=50 --num_experts=1 --zca
python buffer_attacker_6_short_50.py --dataset=CIFAR10 --model=ConvNetD2 --train_epochs=50 --num_experts=1 --zca
python buffer_attacker_6_short_50.py --dataset=CIFAR100 --model=ResNet18 --train_epochs=50 --num_experts=1 --zca
python buffer_attacker_6_short_50.py --dataset=CIFAR100 --model=ResNet34 --train_epochs=50 --num_experts=1 --zca


echo "Buffer Generation Completed"



