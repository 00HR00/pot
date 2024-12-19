# Towards Understanding and Enhancing Security of Proof-of-Training for DNN Model Ownership Verification



## Requirements
python version = 3.9.7
pytorch version = 1.11.0
We use the NVIDIA A100 80GB to run the program.


## Setup
Install the environments
```
conda env create -f env_build.yaml -n pot
```
Activate the environtment
```
conda activate pot
```
If you encounter issues with the installation of certain packages during the setup of your environment, you may initially attempt to resolve these by modifying the `.condarc` file to switch channels. Should this approach fail to rectify the issue, you could consider removing non-essential packages that are causing installation difficulties from the `.yaml` file.


## Generating Trajectories
```
chmod +x buffer_generation_command.sh
```
```
./buffer_generation_command.sh
```
Please ensure that your device has adequate storage capacity, as the volume of data generated is substantial.

## Carrying Out the Distillation Process.
### CIFAR-10
```
chmod +x distill_command_cifar10.sh
```
```
./distill_command_cifar10.sh
```
### CIFAR-100
```
chmod +x distill_command_cifar100.sh
```
```
./distill_command_cifar100.sh
```

## Plotting experimental results
### CIFAR-10
```
python plot_cifar10.py
```
### CIFAR-100
```
python plot_cifar100.py
```


## Acknowledgments
The code is developed based on the TESLA codebases. We deeply appreciate their efforts.
* [Scaling Up Dataset Distillation to ImageNet-1K with Constant Memory (TESLA)](https://github.com/justincui03/tesla)



<!-- # Reference

``` -->




