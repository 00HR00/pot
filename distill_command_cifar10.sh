#!/bin/bash

echo "Starting the CIFAR10 Distillation"



python distill_honest.py  --model=ResNet34  --dataset=CIFAR10 --ipc=10 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_honest.py  --model=ResNet34  --dataset=CIFAR10 --ipc=50 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_honest.py  --model=ResNet18  --dataset=CIFAR10 --ipc=1 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_honest.py  --model=ResNet18  --dataset=CIFAR10 --ipc=10 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_honest.py  --model=ResNet18  --dataset=CIFAR10 --ipc=50 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_honest.py --model=ConvNetD2  --dataset=CIFAR10 --ipc=1 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_honest.py  --model=ConvNetD2  --dataset=CIFAR10 --ipc=10 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_honest.py  --model=ConvNetD2  --dataset=CIFAR10 --ipc=50 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise

python distill_attacker.py --attacker_num=1  --model=ResNet34  --dataset=CIFAR10 --ipc=10 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=1  --model=ResNet34  --dataset=CIFAR10 --ipc=50 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=1  --model=ResNet18  --dataset=CIFAR10 --ipc=1 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=1  --model=ResNet18  --dataset=CIFAR10 --ipc=10 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=1  --model=ResNet18  --dataset=CIFAR10 --ipc=50 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=1  --model=ConvNetD2  --dataset=CIFAR10 --ipc=1 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=1  --model=ConvNetD2  --dataset=CIFAR10 --ipc=10 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=1  --model=ConvNetD2  --dataset=CIFAR10 --ipc=50 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise

python distill_attacker.py --attacker_num=2  --model=ResNet34  --dataset=CIFAR10 --ipc=10 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=2  --model=ResNet34  --dataset=CIFAR10 --ipc=50 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=2  --model=ResNet18  --dataset=CIFAR10 --ipc=1 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=2  --model=ResNet18  --dataset=CIFAR10 --ipc=10 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=2  --model=ResNet18  --dataset=CIFAR10 --ipc=50 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=2  --model=ConvNetD2  --dataset=CIFAR10 --ipc=1 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=2  --model=ConvNetD2  --dataset=CIFAR10 --ipc=10 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=2  --model=ConvNetD2  --dataset=CIFAR10 --ipc=50 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise

python distill_attacker.py --attacker_num=22  --model=ResNet34  --dataset=CIFAR10 --ipc=10 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=22  --model=ResNet34  --dataset=CIFAR10 --ipc=50 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=22  --model=ResNet18  --dataset=CIFAR10 --ipc=1 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=22  --model=ResNet18  --dataset=CIFAR10 --ipc=10 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=22  --model=ResNet18  --dataset=CIFAR10 --ipc=50 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=22  --model=ConvNetD2  --dataset=CIFAR10 --ipc=1 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=22  --model=ConvNetD2  --dataset=CIFAR10 --ipc=10 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=22  --model=ConvNetD2  --dataset=CIFAR10 --ipc=50 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise

python distill_attacker.py --attacker_num=5  --model=ResNet34  --dataset=CIFAR10 --ipc=10 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=5  --model=ResNet34  --dataset=CIFAR10 --ipc=50 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=5  --model=ResNet18  --dataset=CIFAR10 --ipc=1 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=5  --model=ResNet18  --dataset=CIFAR10 --ipc=10 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=5  --model=ResNet18  --dataset=CIFAR10 --ipc=50 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=5  --model=ConvNetD2  --dataset=CIFAR10 --ipc=1 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=5  --model=ConvNetD2  --dataset=CIFAR10 --ipc=10 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=5  --model=ConvNetD2  --dataset=CIFAR10 --ipc=50 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise

python distill_attacker.py --attacker_num=55  --model=ResNet34  --dataset=CIFAR10 --ipc=10 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=55  --model=ResNet34  --dataset=CIFAR10 --ipc=50 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=55  --model=ResNet18  --dataset=CIFAR10 --ipc=1 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=55  --model=ResNet18  --dataset=CIFAR10 --ipc=10 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=55  --model=ResNet18  --dataset=CIFAR10 --ipc=50 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=55  --model=ConvNetD2  --dataset=CIFAR10 --ipc=1 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=55  --model=ConvNetD2  --dataset=CIFAR10 --ipc=10 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=55  --model=ConvNetD2  --dataset=CIFAR10 --ipc=50 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise

python distill_attacker.py --attacker_num=6  --model=ResNet34  --dataset=CIFAR10 --ipc=10 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=6  --model=ResNet34  --dataset=CIFAR10 --ipc=50 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=6  --model=ResNet18  --dataset=CIFAR10 --ipc=1 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=6  --model=ResNet18  --dataset=CIFAR10 --ipc=10 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=6  --model=ResNet18  --dataset=CIFAR10 --ipc=50 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=6  --model=ConvNetD2  --dataset=CIFAR10 --ipc=1 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=6  --model=ConvNetD2  --dataset=CIFAR10 --ipc=10 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=6  --model=ConvNetD2  --dataset=CIFAR10 --ipc=50 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise

python distill_attacker.py --attacker_num=66  --model=ResNet34  --dataset=CIFAR10 --ipc=10 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=66  --model=ResNet34  --dataset=CIFAR10 --ipc=50 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=66  --model=ResNet18  --dataset=CIFAR10 --ipc=1 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=66  --model=ResNet18  --dataset=CIFAR10 --ipc=10 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=66  --model=ResNet18  --dataset=CIFAR10 --ipc=50 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=66 --model=ConvNetD2  --dataset=CIFAR10 --ipc=1 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=66  --model=ConvNetD2  --dataset=CIFAR10 --ipc=10 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=66  --model=ConvNetD2  --dataset=CIFAR10 --ipc=50 --syn_steps=50 --expert_epochs=2 --max_start_epoch=45 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise

echo "CIFAR 10 Distillation Completed"
