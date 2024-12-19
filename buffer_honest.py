import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import get_dataset, get_network, get_resnet_network, get_resnet34_network, get_daparam, get_network_last_iteratioin, get_flatten_parameter, \
    TensorDataset, epoch, ParamDiffAug, epoch_attacker,get_network_last_iteratioin_cifa10, get_network_last_iteratioin_test_convergence, get_target_M_network, l2_norm_similarity, attacker_get_network_last_iteratioin
import copy
import math
import wandb
import time

from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# os.environ["WANDB_MODE"] = "offline"
min_loss1 = math.inf
min_loss2 = math.inf
min_loss3 = math.inf


def main(args):


    wandb.init(sync_tensorboard=False,
               project="TeacherModel",
               job_type="Train",
               name="honest  model={0} dataset={1}".format(args.model, args.dataset),
               config=args,
               )
    

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    start_time = time.time()
    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args=args)

    print('Hyper-parameters: \n', args.__dict__)

    if args.dataset == 'CIFAR10':
        if args.model == 'ConvNet':
            teacher_net_save_path = os.path.join(args.intermediate_result_save_path, "honest_cifar10_ConvNet_t")
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "honest_cifar10_ConvNet_b")
        elif args.model == 'ConvNetD2':
            teacher_net_save_path = os.path.join(args.intermediate_result_save_path, "honest_cifar10_ConvNetD2_t")
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "honest_cifar10_ConvNetD2_b")
        elif args.model == 'ConvNetD4':
            teacher_net_save_path = os.path.join(args.intermediate_result_save_path, "honest_cifar10_ConvNetD4_t")
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "honest_cifar10_ConvNetD4_b")
        elif args.model == 'ResNet18':
            teacher_net_save_path = os.path.join(args.intermediate_result_save_path, "honest_cifar10_ResNet18_t")
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "honest_cifar10_ResNet18_b")
        elif args.model == 'ResNet34':
            teacher_net_save_path = os.path.join(args.intermediate_result_save_path, "honest_cifar10_ResNet34_t")
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "honest_cifar10_ResNet34_b")     
    elif args.dataset == 'CIFAR100':
        if args.model == 'ConvNet':
            teacher_net_save_path = os.path.join(args.intermediate_result_save_path, "honest_cifar100_ConvNet_t")
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "honest_cifar100_ConvNet_b")
        elif args.model == 'ConvNetD2':
            teacher_net_save_path = os.path.join(args.intermediate_result_save_path, "honest_cifar100_ConvNetD2_t")
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "honest_cifar100_ConvNetD2_b")
        elif args.model == 'ConvNetD4':
            teacher_net_save_path = os.path.join(args.intermediate_result_save_path, "honest_cifar100_ConvNetD4_t")
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "honest_cifar100_ConvNetD4_b")
        elif args.model == 'ResNet18':
            teacher_net_save_path = os.path.join(args.intermediate_result_save_path, "honest_cifar100_ResNet18_t")
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "honest_cifar100_ResNet18_b")
        elif args.model == 'ResNet34':
            teacher_net_save_path = os.path.join(args.intermediate_result_save_path, "honest_cifar100_ResNet34_t")
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "honest_cifar100_ResNet34_b")

    # emd
    emd_path = os.path.join(args.intermediate_result_save_path, "emd_honest")
    if not os.path.exists(emd_path):
        os.makedirs(emd_path)

    save_dir = buffer_s_data_dir
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(teacher_net_save_path):
        os.makedirs(teacher_net_save_path)

    if args.loss_type == 'ce':
        criterion = nn.CrossEntropyLoss().to(args.device)
    else:
        criterion = nn.MSELoss().to(args.device)

    criterion_ = nn.MSELoss(reduction='sum').to(args.device)

    trajectories = []

    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    ''' set augmentation for whole-dataset training '''
    args.dc_aug_param = get_daparam(args.dataset, args.model, args.model, None)
    args.dc_aug_param['strategy'] = 'crop_scale_rotate'  
    print('DC augmentation parameters: \n', args.dc_aug_param)
    last_num = 0
    flag = 0

    for it in range(0, args.num_experts):

        ''' Train synthetic data '''
        if it == 0:
            if args.model == 'ConvNet':
                teacher_net = get_network(args.model, channel, num_classes, im_size).to(args.device)
            elif args.model == 'ConvNetD2':
                teacher_net = get_network(args.model, channel, num_classes, im_size).to(args.device)
                # emd
                torch.save(teacher_net, os.path.join(emd_path, "teacher_net_{}.pt".format(0)))
            elif args.model == 'ConvNetD4':
                teacher_net = get_network(args.model, channel, num_classes, im_size).to(args.device)
            elif args.model == 'ResNet18':
                teacher_net = get_resnet_network(args.model, channel, num_classes, im_size).to(args.device)
            elif args.model == 'ResNet34':
                teacher_net = get_resnet34_network(args.model, channel, num_classes, im_size).to(args.device)

            torch.save(teacher_net, os.path.join(teacher_net_save_path, "teacher_net_{}.pt".format(0)))
            
        else:
            teacher_net = get_network_last_iteratioin_cifa10(last_num,teacher_net_save_path).to(args.device)

        teacher_net.train()
        lr = args.lr_teacher
        teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)
        teacher_optim.zero_grad()

        timestamps = []

        lr_schedule = [args.train_epochs // 2 + 1]
        
        for e in range(args.train_epochs):
            if last_num==0:
                timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])
            if last_num==(args.train_epochs-1) and flag!=1:
                flag = 1
                continue

            train_loss, train_acc = epoch("train", dataloader=trainloader, net=teacher_net, optimizer=teacher_optim,
                                        criterion=criterion, args=args, aug=True)

            test_loss, test_acc = epoch("test", dataloader=testloader, net=teacher_net, optimizer=None,
                                        criterion=criterion, args=args, aug=False)
            step_w = it * args.train_epochs + e + 1
            wandb.log({"Test_acc": test_acc}, step=step_w)
            wandb.log({"Test_loss": test_loss}, step=step_w)
            print("Itr: {}\tEpoch: {}\tTrain Acc: {}\tTest Acc: {}".format(it, e, train_acc, test_acc))
            print("Itr: {}\tEpoch: {}\tTrain Acc: {}".format(it, e, train_acc))

            if last_num+1==49 or last_num+1==99 or last_num+1==149:
                torch.save(teacher_net, os.path.join(teacher_net_save_path, "teacher_net_{}.pt".format(last_num+1)))

            #emd
            if args.model == 'ConvNetD2':
                torch.save(teacher_net, os.path.join(emd_path, "teacher_net_{}.pt".format(last_num+1)))

            last_num = last_num + 1

            timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

            if e in lr_schedule and args.decay:
                lr *= 0.1
                teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)
                teacher_optim.zero_grad()

        trajectories.append(timestamps)

        if len(trajectories) == args.save_interval:
            n = 0
            while os.path.exists(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))):
                n += 1

            torch.save(trajectories, os.path.join(save_dir, "replay_buffer_{}.pt".format(n)))
            
            trajectories = []
    end_time = time.time()
    execution_time = end_time - start_time
    txt_dir = './txt/'
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)
    file_name = "honest_training_efficiency.txt"
    e_txt = open(txt_dir+file_name, 'a')
    e_txt.write("{0}    {1}    {2}".format(execution_time,args.model,args.dataset)+"\n")
    e_txt.close()
    print(f"time:{execution_time}s    {args.model} {args.dataset}")
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='dataset')
    parser.add_argument('--model', type=str, default='ResNet34', help='model')
    parser.add_argument('--num_experts', type=int, default=3, help='training iterations')

    parser.add_argument('--lr_teacher', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real loader')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='./data', help='dataset path')

    parser.add_argument('--intermediate_result_save_path', type=str, default='/root/intermediate_result', help='buffer path')
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--zca', action='store_true', default=False)
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--mom', type=float, default=0, help='momentum')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization')
    parser.add_argument('--loss_type', type=str, default='ce', help='loss type, ce or mse')
    parser.add_argument('--save_interval', type=int, default=1)

    parser.add_argument('--alpha_para', type=float, default=0.01, help='alpha')

    parser.add_argument('--teacher_label', action='store_true', default=False, help='whether to use label from the expert model to guide the distillation process.')
    args = parser.parse_args()
    main(args)

