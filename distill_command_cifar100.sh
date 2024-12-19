#!/bin/bash

echo "Starting the CIFAR100 Distillation"


python distill_honest.py  --model=ResNet34  --dataset=CIFAR100 --ipc=10 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_honest.py  --model=ResNet34  --dataset=CIFAR100 --ipc=5 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_honest.py  --model=ResNet18  --dataset=CIFAR100 --ipc=1 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_honest.py  --model=ResNet18  --dataset=CIFAR100 --ipc=10 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_honest.py  --model=ResNet18  --dataset=CIFAR100 --ipc=50 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_honest.py --model=ConvNetD2  --dataset=CIFAR100 --ipc=1 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_honest.py  --model=ConvNetD2  --dataset=CIFAR100 --ipc=10 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_honest.py  --model=ConvNetD2  --dataset=CIFAR100 --ipc=50 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise

python distill_attacker.py --attacker_num=1  --model=ResNet34  --dataset=CIFAR100 --ipc=10 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=1  --model=ResNet34  --dataset=CIFAR100 --ipc=5 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=1  --model=ResNet18  --dataset=CIFAR100 --ipc=1 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=1  --model=ResNet18  --dataset=CIFAR100 --ipc=10 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=1  --model=ResNet18  --dataset=CIFAR100 --ipc=50 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=1  --model=ConvNetD2  --dataset=CIFAR100 --ipc=1 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=1  --model=ConvNetD2  --dataset=CIFAR100 --ipc=10 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=1  --model=ConvNetD2  --dataset=CIFAR100 --ipc=50 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise

python distill_attacker.py --attacker_num=2  --model=ResNet34  --dataset=CIFAR100 --ipc=10 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=2  --model=ResNet34  --dataset=CIFAR100 --ipc=5 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=2  --model=ResNet18  --dataset=CIFAR100 --ipc=1 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=2  --model=ResNet18  --dataset=CIFAR100 --ipc=10 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=2  --model=ResNet18  --dataset=CIFAR100 --ipc=50 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=2  --model=ConvNetD2  --dataset=CIFAR100 --ipc=1 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=2  --model=ConvNetD2  --dataset=CIFAR100 --ipc=10 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=2  --model=ConvNetD2  --dataset=CIFAR100 --ipc=50 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise

python distill_attacker.py --attacker_num=5  --model=ResNet34  --dataset=CIFAR100 --ipc=10 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=5  --model=ResNet34  --dataset=CIFAR100 --ipc=5 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=5  --model=ResNet18  --dataset=CIFAR100 --ipc=1 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=5  --model=ResNet18  --dataset=CIFAR100 --ipc=10 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=5  --model=ResNet18  --dataset=CIFAR100 --ipc=50 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=5  --model=ConvNetD2  --dataset=CIFAR100 --ipc=1 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=5  --model=ConvNetD2  --dataset=CIFAR100 --ipc=10 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=5  --model=ConvNetD2  --dataset=CIFAR100 --ipc=50 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise

python distill_attacker.py --attacker_num=6  --model=ResNet34  --dataset=CIFAR100 --ipc=10 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=6  --model=ResNet34  --dataset=CIFAR100 --ipc=5 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=6  --model=ResNet18  --dataset=CIFAR100 --ipc=1 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=6  --model=ResNet18  --dataset=CIFAR100 --ipc=10 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=6  --model=ResNet18  --dataset=CIFAR100 --ipc=50 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=6  --model=ConvNetD2  --dataset=CIFAR100 --ipc=1 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=6  --model=ConvNetD2  --dataset=CIFAR100 --ipc=10 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=6  --model=ConvNetD2  --dataset=CIFAR100 --ipc=50 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise

python distill_attacker.py --attacker_num=22  --model=ResNet34  --dataset=CIFAR100 --ipc=10 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=22  --model=ResNet34  --dataset=CIFAR100 --ipc=5 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=22  --model=ResNet18  --dataset=CIFAR100 --ipc=1 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=22  --model=ResNet18  --dataset=CIFAR100 --ipc=10 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=22  --model=ResNet18  --dataset=CIFAR100 --ipc=50 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=22  --model=ConvNetD2  --dataset=CIFAR100 --ipc=1 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=22  --model=ConvNetD2  --dataset=CIFAR100 --ipc=10 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=22  --model=ConvNetD2  --dataset=CIFAR100 --ipc=50 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise

python distill_attacker.py --attacker_num=55  --model=ResNet34  --dataset=CIFAR100 --ipc=10 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=55  --model=ResNet34  --dataset=CIFAR100 --ipc=5 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=55  --model=ResNet18  --dataset=CIFAR100 --ipc=1 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=55  --model=ResNet18  --dataset=CIFAR100 --ipc=10 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=55  --model=ResNet18  --dataset=CIFAR100 --ipc=50 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=55  --model=ConvNetD2  --dataset=CIFAR100 --ipc=1 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=55  --model=ConvNetD2  --dataset=CIFAR100 --ipc=10 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=55  --model=ConvNetD2  --dataset=CIFAR100 --ipc=50 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise

python distill_attacker.py --attacker_num=66  --model=ResNet34  --dataset=CIFAR100 --ipc=10 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=66  --model=ResNet34  --dataset=CIFAR100 --ipc=5 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=66  --model=ResNet18  --dataset=CIFAR100 --ipc=1 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=66  --model=ResNet18  --dataset=CIFAR100 --ipc=10 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=66  --model=ResNet18  --dataset=CIFAR100 --ipc=50 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=66 --model=ConvNetD2  --dataset=CIFAR100 --ipc=1 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=66  --model=ConvNetD2  --dataset=CIFAR100 --ipc=10 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise
python distill_attacker.py --attacker_num=66  --model=ConvNetD2  --dataset=CIFAR100 --ipc=50 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-07 --lr_teacher=0.01 --Iteration=310 --pix_init=noise

echo "CIFAR100 Distillation Completed"
