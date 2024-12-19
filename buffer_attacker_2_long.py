import torch
import torch.nn as nn
import torch 
import argparse
import copy
import os
import time
from utils import get_dataset, get_network, get_resnet_network, get_resnet34_network, get_daparam, get_flatten_parameter, \
    TensorDataset, epoch, ParamDiffAug, epoch_attacker_to_m0, get_initial_M_network, get_target_M_network, l2_norm_similarity, attacker_get_network_last_iteratioin



def generate_and_save_models(n, save_dir, initial_model, model_A, model_B):
    models = []
    timestamps = []
    trajectories = []
    num = 0
    for i in range(n):
        model_g = initial_model
        for param_key in model_A.state_dict():
            param_A = model_A.state_dict()[param_key]
            param_B = model_B.state_dict()[param_key]
            model_g.state_dict()[param_key].copy_((1 - i/(n-1)) * param_A + (i/(n-1)) * param_B)
        models.append(model_g)

        
        timestamps.append([p.detach().cpu() for p in model_g.parameters()])
        if (i+1) % 50 == 0:
            trajectories.append(timestamps)
            torch.save(trajectories, os.path.join(save_dir, "replay_buffer_{}.pt".format(num)))
            num  += 1

            trajectories = []
            timestamps = []

    return models



def main(args):
    start_time = time.time()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'



    target_path = ''
    buffer_s_data_dir = ''
    target_M_network = ''
    initial_M_network = ''
    honest_length = args.target_model_num

    if args.dataset == 'CIFAR10':
        if args.model == 'ConvNet':
            target_path = os.path.join(args.intermediate_result_save_path, "honest_cifar10_ConvNet_t")
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "attacker_22_cifar10_ConvNet_b")
            target_M_network = get_target_M_network(honest_length,target_path).to(args.device)
            initial_M_network = get_initial_M_network(0,target_path).to(args.device)

        elif args.model == 'ConvNetD2':
            target_path = os.path.join(args.intermediate_result_save_path, "honest_cifar10_ConvNetD2_t")
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "attacker_22_cifar10_ConvNetD2_b")
            target_M_network = get_target_M_network(honest_length,target_path).to(args.device)
            initial_M_network = get_initial_M_network(0,target_path).to(args.device)

        elif args.model == 'ConvNetD4':
            target_path = os.path.join(args.intermediate_result_save_path, "honest_cifar10_ConvNetD4_t")
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "attacker_22_cifar10_ConvNetD4_b")
            target_M_network = get_target_M_network(honest_length,target_path).to(args.device)
            initial_M_network = get_initial_M_network(0,target_path).to(args.device)
                        
        elif args.model == 'ResNet18':
            target_path = os.path.join(args.intermediate_result_save_path, "honest_cifar10_ResNet18_t")
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "attacker_22_cifar10_ResNet18_b")
            target_M_network = get_target_M_network(honest_length,target_path).to(args.device)
            initial_M_network = get_initial_M_network(0,target_path).to(args.device)
            
        elif args.model == 'ResNet34':
            target_path = os.path.join(args.intermediate_result_save_path, "honest_cifar10_ResNet34_t")
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "attacker_22_cifar10_ResNet34_b")
            target_M_network = get_target_M_network(honest_length,target_path).to(args.device)
            initial_M_network = get_initial_M_network(0,target_path).to(args.device)
      
    elif args.dataset == 'CIFAR100':
        if args.model == 'ConvNet':
            target_path = os.path.join(args.intermediate_result_save_path, "honest_cifar100_ConvNet_t")
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "attacker_22_cifar100_ConvNet_b")
            target_M_network = get_target_M_network(honest_length,target_path).to(args.device)
            initial_M_network = get_initial_M_network(0,target_path).to(args.device)

        elif args.model == 'ConvNetD2':
            target_path = os.path.join(args.intermediate_result_save_path, "honest_cifar100_ConvNetD2_t")
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "attacker_22_cifar100_ConvNetD2_b")
            target_M_network = get_target_M_network(honest_length,target_path).to(args.device)
            initial_M_network = get_initial_M_network(0,target_path).to(args.device)

        elif args.model == 'ConvNetD4':
            target_path = os.path.join(args.intermediate_result_save_path, "honest_cifar100_ConvNetD4_t")
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "attacker_22_cifar100_ConvNetD4_b")
            target_M_network = get_target_M_network(honest_length,target_path).to(args.device)
            initial_M_network = get_initial_M_network(0,target_path).to(args.device)
       
        elif args.model == 'ResNet18':
            target_path = os.path.join(args.intermediate_result_save_path, "honest_cifar100_ResNet18_t")
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "attacker_22_cifar100_ResNet18_b")
            target_M_network = get_target_M_network(honest_length,target_path).to(args.device)
            initial_M_network = get_initial_M_network(0,target_path).to(args.device)
            
        elif args.model == 'ResNet34':
            target_path = os.path.join(args.intermediate_result_save_path, "honest_cifar100_ResNet34_t")
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "attacker_22_cifar100_ResNet34_b")
            target_M_network = get_target_M_network(honest_length,target_path).to(args.device)
            initial_M_network = get_initial_M_network(0,target_path).to(args.device)

    channel = 3
    im_size = (32, 32)
    num_classes = 0
    if args.dataset == "CIFAR10":
        num_classes = 10
    elif args.dataset == "CIFAR100":
        num_classes = 100

    if args.model == 'ConvNet':
        initial_M_network = get_network(args.model, channel, num_classes, im_size).to(args.device)
    elif args.model == 'ConvNetD2':
        initial_M_network = get_network(args.model, channel, num_classes, im_size).to(args.device)
    elif args.model == 'ConvNetD4':
        initial_M_network = get_network(args.model, channel, num_classes, im_size).to(args.device)
    elif args.model == 'ResNet18':
        initial_M_network = get_resnet_network(args.model, channel, num_classes, im_size).to(args.device)
    elif args.model == 'ResNet34':
        initial_M_network = get_resnet34_network(args.model, channel, num_classes, im_size).to(args.device)

    model_A = copy.deepcopy(initial_M_network)
    initial_model_ = copy.deepcopy(initial_M_network)
    model_B = copy.deepcopy(target_M_network)

    print(args)
    n = args.n_length
    save_directory = buffer_s_data_dir  
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    generated_models = generate_and_save_models(n, save_directory, initial_model=initial_model_, model_A=model_A, model_B=model_B)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"time:{execution_time}s    {args.model} {args.dataset}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--n_length', type=float, default=250, help='trajectory_length')
    parser.add_argument('--alpha_para', type=float, default=0.3, help='alpha')
    parser.add_argument('--target_model_num', type=int, default=149, help='target model num')

    parser.add_argument('--model', type=str, default='', help='model')
    parser.add_argument('--dataset', type=str, default='', help='dataset')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real loader')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--intermediate_result_save_path', type=str, default='/root/intermediate_result', help='dataset path')



    args = parser.parse_args()
    main(args)