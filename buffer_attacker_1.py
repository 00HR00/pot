import torch 

import argparse
import os
from utils import get_dataset, get_network, get_resnet_network, get_resnet34_network, get_daparam, get_flatten_parameter, \
    TensorDataset, epoch, ParamDiffAug, epoch_attacker_to_m0, get_initial_M_network, get_target_M_network, l2_norm_similarity, attacker_get_network_last_iteratioin


import copy
import time
def update_model(model_A, model_B, update_fraction, num_updates, save_path):

    model_A.eval()
    model_B.eval()


    weight_diff = []
    for (param_A, param_B) in zip(model_A.parameters(), model_B.parameters()):
        weight_diff.append(param_B.data - param_A.data)


    timestamps = []
    trajectories = []
    num = 0
    for update in range(num_updates):

        for param, diff in zip(model_A.parameters(), weight_diff):
            param.data += diff * update_fraction

        timestamps.append([p.detach().cpu() for p in model_A.parameters()])

        if (update+1) % 50 == 0:
            trajectories.append(timestamps)

            torch.save(trajectories, os.path.join(save_path, "replay_buffer_{}.pt".format(num)))
            num += 1

            trajectories = []
            timestamps = []



def main(args):
    start_time = time.time()

    trajectories = []

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    target_path = ''
    buffer_s_data_dir = ''
    target_M_network = ''
    initial_M_network = ''
    honest_length = args.target_num

    if args.dataset == 'CIFAR10':
        if args.model == 'ConvNet':
            target_path = os.path.join(args.intermediate_result_save_path, "honest_cifar10_ConvNet_t")
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "attacker_1_cifar10_ConvNet_b")
            target_M_network = get_target_M_network(honest_length,target_path).to(args.device)
            initial_M_network = get_initial_M_network(0,target_path).to(args.device)

        elif args.model == 'ConvNetD2':
            target_path = os.path.join(args.intermediate_result_save_path, "honest_cifar10_ConvNetD2_t")
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "attacker_1_cifar10_ConvNetD2_b")
            target_M_network = get_target_M_network(honest_length,target_path).to(args.device)
            initial_M_network = get_initial_M_network(0,target_path).to(args.device)

        elif args.model == 'ConvNetD4':
            target_path = os.path.join(args.intermediate_result_save_path, "honest_cifar10_ConvNetD4_t")
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "attacker_1_cifar10_ConvNetD4_b")
            target_M_network = get_target_M_network(honest_length,target_path).to(args.device)
            initial_M_network = get_initial_M_network(0,target_path).to(args.device)

        elif args.model == 'ResNet18':
            target_path = os.path.join(args.intermediate_result_save_path, "honest_cifar10_ResNet18_t")
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "attacker_1_cifar10_ResNet18_b")
            target_M_network = get_target_M_network(honest_length,target_path).to(args.device)
            initial_M_network = get_initial_M_network(0,target_path).to(args.device)
            
        elif args.model == 'ResNet34':
            target_path = os.path.join(args.intermediate_result_save_path, "honest_cifar10_ResNet34_t")
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "attacker_1_cifar10_ResNet34_b")
            target_M_network = get_target_M_network(honest_length,target_path).to(args.device)
            initial_M_network = get_initial_M_network(0,target_path).to(args.device)
      
    elif args.dataset == 'CIFAR100':
        if args.model == 'ConvNet':
            target_path = os.path.join(args.intermediate_result_save_path, "honest_cifar100_ConvNet_t")
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "attacker_1_cifar100_ConvNet_b")
            target_M_network = get_target_M_network(honest_length,target_path).to(args.device)
            initial_M_network = get_initial_M_network(0,target_path).to(args.device)

        elif args.model == 'ConvNetD2':
            target_path = os.path.join(args.intermediate_result_save_path, "honest_cifar100_ConvNetD2_t")
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "attacker_1_cifar100_ConvNetD2_b")
            target_M_network = get_target_M_network(honest_length,target_path).to(args.device)
            initial_M_network = get_initial_M_network(0,target_path).to(args.device)

        elif args.model == 'ConvNetD4':
            target_path = os.path.join(args.intermediate_result_save_path, "honest_cifar100_ConvNetD4_t")
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "attacker_1_cifar100_ConvNetD4_b")
            target_M_network = get_target_M_network(honest_length,target_path).to(args.device)
            initial_M_network = get_initial_M_network(0,target_path).to(args.device)
            
        elif args.model == 'ResNet18':
            target_path = os.path.join(args.intermediate_result_save_path, "honest_cifar100_ResNet18_t")
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "attacker_1_cifar100_ResNet18_b")
            target_M_network = get_target_M_network(honest_length,target_path).to(args.device)
            initial_M_network = get_initial_M_network(0,target_path).to(args.device)
            
        elif args.model == 'ResNet34':
            target_path = os.path.join(args.intermediate_result_save_path, "honest_cifar100_ResNet34_t")
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "attacker_1_cifar100_ResNet34_b")
            target_M_network = get_target_M_network(honest_length,target_path).to(args.device)
            initial_M_network = get_initial_M_network(0,target_path).to(args.device)



    initial_M_network.eval() 
    target_M_network.eval() 


    channel = 3
    im_size = (32, 32)
    num_classes = 0
    if args.dataset == "CIFAR10":
        num_classes = 10
    elif args.dataset == "CIFAR100":
        num_classes = 100


    if args.model == 'ConvNet':
        model_A = get_network(args.model, channel, num_classes, im_size).to(args.device)
    elif args.model == 'ConvNetD2':
        model_A = get_network(args.model, channel, num_classes, im_size).to(args.device)
    elif args.model == 'ConvNetD4':
        model_A = get_network(args.model, channel, num_classes, im_size).to(args.device)
    elif args.model == 'ResNet18':
        model_A = get_resnet_network(args.model, channel, num_classes, im_size).to(args.device)
    elif args.model == 'ResNet34':
        model_A = get_resnet34_network(args.model, channel, num_classes, im_size).to(args.device) 
    model_B = target_M_network 
    
    num_updates = args.times
    update_fraction = 1/num_updates 

    save_path = buffer_s_data_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    update_model(model_A, model_B, update_fraction, num_updates, save_path)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"time:{execution_time}s    {args.model} {args.dataset}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--times', type=float, default=150, help='divison_factor')
    parser.add_argument('--alpha_para', type=float, default=0.3, help='alpha')
    parser.add_argument('--target_num', type=int, default=149, help='alpha')

    parser.add_argument('--model', type=str, default='', help='model')
    parser.add_argument('--dataset', type=str, default='', help='dataset')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real loader')
    parser.add_argument('--intermediate_result_save_path', type=str, default='/root/intermediate_result', help='dataset path')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='./intermediate_result/attacker_1_s_data_train_buffer_cifar10_ConvNet/CIFAR10/ConvNet', help='dataset path')

    args = parser.parse_args()
    print(args)
    main(args)