import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from utils import augment, get_dataset, get_network, get_resnet34_network, get_resnet_network, get_eval_pool, evaluate_synset, evaluate_test_mn_synset, get_time, DiffAugment, DiffAugmentList, ParamDiffAug
import wandb
import copy
import random
import time
from reparam_module import ReparamModule
from torch.utils.data import Subset
from torch.utils.data import DataLoader

from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# os.environ["WANDB_MODE"] = "offline"
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def main(args):
    start_time = time.time()


    buffer_s_data_dir = ''
    save_path = ''
    max_acc = 0
    acc_list = []
    

    if args.dataset == 'CIFAR10':
        if args.model == 'ConvNet':
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "attacker_" + str(args.attacker_num) + "_cifar10_ConvNet_b")
            save_path = os.path.join(args.syn_image_save_path, "attacker_" + str(args.attacker_num) + "_cifar10_ConvNet_syn")
            save_path = save_path + "_"+str(args.ipc)
        elif args.model == 'ConvNetD2':
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "attacker_" + str(args.attacker_num) + "_cifar10_ConvNetD2_b")
            save_path = os.path.join(args.syn_image_save_path, "attacker_" + str(args.attacker_num) + "_cifar10_ConvNetD2_syn")
            save_path = save_path + "_"+str(args.ipc)
        elif args.model == 'ConvNetD4':
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "attacker_" + str(args.attacker_num) + "_cifar10_ConvNetD4_b")
            save_path = os.path.join(args.syn_image_save_path, "attacker_" + str(args.attacker_num) + "_cifar10_ConvNetD4_syn")
            save_path = save_path + "_"+str(args.ipc)
        elif args.model == 'ResNet18':
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "attacker_" + str(args.attacker_num) + "_cifar10_ResNet18_b")
            save_path = os.path.join(args.syn_image_save_path, "attacker_" + str(args.attacker_num) + "_cifar10_ResNet18_syn")
            save_path = save_path + "_"+str(args.ipc)
        elif args.model == 'ResNet34':
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "attacker_" + str(args.attacker_num) + "_cifar10_ResNet34_b")
            save_path = os.path.join(args.syn_image_save_path, "attacker_" + str(args.attacker_num) + "_cifar10_ResNet34_syn")
            save_path = save_path + "_"+str(args.ipc)
    elif args.dataset == 'CIFAR100':
        if args.model == 'ConvNet':
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "attacker_" + str(args.attacker_num) + "_cifar100_ConvNet_b")
            save_path = os.path.join(args.syn_image_save_path, "attacker_" + str(args.attacker_num) + "_cifar100_ConvNet_syn")
            save_path = save_path + "_"+str(args.ipc)
        elif args.model == 'ConvNetD2':
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "attacker_" + str(args.attacker_num) + "_cifar100_ConvNetD2_b")
            save_path = os.path.join(args.syn_image_save_path, "attacker_" + str(args.attacker_num) + "_cifar100_ConvNetD2_syn")
            save_path = save_path + "_"+str(args.ipc)
        elif args.model == 'ConvNetD4':
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "attacker_" + str(args.attacker_num) + "_cifar100_ConvNetD4_b")
            save_path = os.path.join(args.syn_image_save_path, "attacker_" + str(args.attacker_num) + "_cifar100_ConvNetD4_syn")
            save_path = save_path + "_"+str(args.ipc)
        elif args.model == 'ResNet18':
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "attacker_" + str(args.attacker_num) + "_cifar100_ResNet18_b")
            save_path = os.path.join(args.syn_image_save_path, "attacker_" + str(args.attacker_num) + "_cifar100_ResNet18_syn")
            save_path = save_path + "_"+str(args.ipc)
        elif args.model == 'ResNet34':
            buffer_s_data_dir = os.path.join(args.intermediate_result_save_path, "attacker_" + str(args.attacker_num) + "_cifar100_ResNet34_b")
            save_path = os.path.join(args.syn_image_save_path, "attacker_" + str(args.attacker_num) + "_cifar100_ResNet34_syn")
            save_path = save_path + "_"+str(args.ipc)


    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()


    if args.zca and args.texture:
        raise AssertionError("Cannot use zca and texture together")

    if args.texture and args.pix_init == "real":
        print("WARNING: Using texture with real initialization will take a very long time to smooth out the boundaries between images.")

    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args=args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    im_res = im_size[0]

    args.im_size = im_size

    accs_all_exps = dict() 
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    if args.dsa:

        args.dc_aug_param = None

    args.dsa_param = ParamDiffAug()

    dsa_params = args.dsa_param
    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None



    wandb.init(sync_tensorboard=False,
               project="DatasetDistillation",
               job_type="CleanRepo",
               name="attacker {0} ipc={1} model={2} dataset={3}".format(args.attacker_num,args.ipc,args.model,args.dataset),
               config=args,
               )

    args = type('', (), {})()

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    if args.batch_syn is None:
        args.batch_syn = num_classes * args.ipc

    args.distributed = torch.cuda.device_count() > 1


    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    ''' organize the real dataset '''
    indices_class = [[] for c in range(num_classes)]

    print("---------------Build label to index map--------------")

    if args.dataset == 'ImageNet':
        indices_class = np.load('indices/imagenet_indices_class.npy', allow_pickle=True)
    elif args.dataset == 'Tiny':
        indices_class = np.load('indices/tiny_indices_class.npy', allow_pickle=True)
    else:
        for i, data in tqdm(enumerate(dst_train)):
            indices_class[data[1]].append(i)



    def get_images(c, n):  
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        subset = Subset(dst_train, idx_shuffle)
        data_loader = DataLoader(subset, batch_size=n)

        for data in data_loader:
            return data[0].to("cpu")


    ''' initialize the synthetic data '''
    label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1)

    if args.texture:
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0]*args.canvas_size, im_size[1]*args.canvas_size), dtype=torch.float)
    else:
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)

    syn_lr = torch.tensor(args.lr_teacher).to(args.device)

    if args.pix_init == 'real':
        print('initialize synthetic data from random real images')
        for c in range(num_classes):
            image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data
    else:
        print('initialize synthetic data from random noise')


    ''' training '''
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    print(image_syn.shape)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)

    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)
    optimizer_img.zero_grad()
    optimizer_lr.zero_grad()

    criterion = nn.CrossEntropyLoss().to(args.device)
    print('%s training begins'%get_time())


    expert_dir = buffer_s_data_dir
    print("Expert Dir: {}".format(expert_dir))

    if not args.random_trajectory:
        if args.load_all:
            buffer = []
            n = 0
            while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
                buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
                n += 1
            if n == 0:
                raise AssertionError("No buffers detected at {}".format(expert_dir))

        else:
            expert_files = []
            n = 0
            while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
                expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
                n += 1
            if n == 0:
                raise AssertionError("No buffers detected at {}".format(expert_dir))
            file_idx = 0
            expert_idx = 0
            random.shuffle(expert_files)
            if args.max_files is not None:
                expert_files = expert_files[:args.max_files]
            print("loading file {}".format(expert_files[file_idx]))
            buffer = torch.load(expert_files[file_idx])
            if args.max_experts is not None:
                buffer = buffer[:args.max_experts]
            random.shuffle(buffer)

    best_acc = {m: 0 for m in model_eval_pool}

    best_std = {m: 0 for m in model_eval_pool}
    txt_dir = './txt/'
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)
    acc_file_name = "distill_acc_attacker_"+str(args.attacker_num)+"_"+str(args.model)+"_"+str(args.dataset)+"_"+str(args.ipc)+".txt"
    e_txt_acc = open(txt_dir+acc_file_name, 'w+')
    for it in range(0, args.Iteration+1):
        save_this_it = False

        wandb.log({"Progress": it}, step=it)
        ''' Evaluate synthetic data '''
        if it in eval_it_pool and args.eval_it > 0:
            for model_eval in model_eval_pool:
                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                if args.dsa:
                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                else:
                    print('DC augmentation parameters: \n', args.dc_aug_param)

                accs_test = []
                accs_train = []
                for it_eval in range(args.num_eval):
                    if args.model=="ResNet34":
                        net_eval = get_resnet34_network(model_eval, channel, num_classes, im_size).to(args.device)
                    elif args.model=="ResNet18":
                        net_eval = get_resnet_network(model_eval, channel, num_classes, im_size).to(args.device) 
                    else:
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) 

                    eval_labs = label_syn
                    with torch.no_grad():
                        image_save = image_syn
                    image_syn_eval, label_syn_eval = copy.deepcopy(image_save.detach()), copy.deepcopy(eval_labs.detach()) 

                    args.lr_net = syn_lr.item()
                    _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, texture=args.texture)
                    accs_test.append(acc_test)
                    accs_train.append(acc_train)
                accs_test = np.array(accs_test)
                accs_train = np.array(accs_train)
                acc_test_mean = np.mean(accs_test)
                acc_test_std = np.std(accs_test)
                if acc_test_mean > best_acc[model_eval]:
                    best_acc[model_eval] = acc_test_mean
                    best_std[model_eval] = acc_test_std
                    save_this_it = True
                print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs_test), model_eval, acc_test_mean, acc_test_std))
                acc_list.append(acc_test_mean)
                e_txt_acc.write("{0}".format(acc_test_mean)+"\n")
                wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean}, step=it)
                wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[model_eval]}, step=it)
                wandb.log({'Std/{}'.format(model_eval): acc_test_std}, step=it)
                wandb.log({'Max_Std/{}'.format(model_eval): best_std[model_eval]}, step=it)


        if it in eval_it_pool and (save_this_it or it % 1000 == 0) and args.eval_it > 0:
            with torch.no_grad():
                image_save = image_syn.cuda()

                save_dir = os.path.join(".", "logged_files", args.dataset, 'offline' if wandb.run.name is None else wandb.run.name)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                torch.save(image_save.cpu(), os.path.join(save_dir, "images_{}.pt".format(it)))
                torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_{}.pt".format(it)))

                if save_this_it:
                    torch.save(image_save.cpu(), os.path.join(save_dir, "images_best.pt".format(it)))
                    torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_best.pt".format(it)))

                wandb.log({"Pixels": wandb.Histogram(torch.nan_to_num(image_syn.detach().cpu()))}, step=it)

                if args.ipc < 50 or args.force_save:
                    upsampled = image_save
                    if args.dataset != "ImageNet":
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                    grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                    wandb.log({"Synthetic_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                    wandb.log({'Synthetic_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)

                    for clip_val in [2.5]:
                        std = torch.std(image_save)
                        mean = torch.mean(image_save)
                        upsampled = torch.clip(image_save, min=mean-clip_val*std, max=mean+clip_val*std)
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"Clipped_Synthetic_Images/std_{}".format(clip_val): wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)

                    if args.zca:
                        image_save = image_save.to(args.device)
                        image_save = args.zca_trans.inverse_transform(image_save)
                        image_save.cpu()

                        torch.save(image_save.cpu(), os.path.join(save_dir, "images_zca_{}.pt".format(it)))

                        upsampled = image_save
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"Reconstructed_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                        wandb.log({'Reconstructed_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)

                        for clip_val in [2.5]:
                            std = torch.std(image_save)
                            mean = torch.mean(image_save)
                            upsampled = torch.clip(image_save, min=mean - clip_val * std, max=mean + clip_val * std)
                            if args.dataset != "ImageNet":
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                            grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                            wandb.log({"Clipped_Reconstructed_Images/std_{}".format(clip_val): wandb.Image(
                                torch.nan_to_num(grid.detach().cpu()))}, step=it)

        wandb.log({"Synthetic_LR": syn_lr.detach().cpu()}, step=it)

        if args.model=="ResNet34":
            student_net = get_resnet34_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)
        elif args.model=="ResNet18":
            student_net = get_resnet_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)
        else:
            student_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)

        student_net = ReparamModule(student_net)

        if args.distributed:
            student_net = torch.nn.DataParallel(student_net)

        student_net.train()

        if not args.random_trajectory:
            if args.load_all:
                expert_trajectory = buffer[np.random.randint(0, len(buffer))]
            else:
                expert_trajectory = buffer[expert_idx]
                expert_idx += 1
                if expert_idx == len(buffer):
                    expert_idx = 0
                    file_idx += 1
                    if file_idx == len(expert_files):
                        file_idx = 0
                        random.shuffle(expert_files)
                    print("loading file {}".format(expert_files[file_idx]))
                    if args.max_files != 1:
                        del buffer
                        buffer = torch.load(expert_files[file_idx])
                    if args.max_experts is not None:
                        buffer = buffer[:args.max_experts]
                    random.shuffle(buffer)

        start_epoch = np.random.randint(0, args.max_start_epoch)

        if not args.random_trajectory:
            starting_params = expert_trajectory[start_epoch]

            target_params = expert_trajectory[start_epoch+args.expert_epochs]
        else:
            starting_params = [p for p in student_net.parameters()]
            target_params = [p for p in student_net.parameters()]

        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)

        student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]

        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)

        param_dist = torch.tensor(0.0).to(args.device)

        param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")


        if args.teacher_label:

            if args.model=="ResNet34":
                label_net = get_resnet34_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)
            elif args.model=="ResNet18":
                label_net = get_resnet_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)
            else:
                label_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device) 

            label_net = ReparamModule(label_net)
            label_net.eval()


            label_params = copy.deepcopy(target_params.detach()).requires_grad_(False)

            batch_labels = []
            SOFT_INIT_BATCH_SIZE = 50
            if image_syn.shape[0] > SOFT_INIT_BATCH_SIZE and args.dataset == 'ImageNet':
                for indices in torch.split(torch.tensor([i for i in range(0, image_syn.shape[0])], dtype=torch.long), SOFT_INIT_BATCH_SIZE):
                    batch_labels.append(label_net(image_syn[indices].detach().to(args.device), flat_param=label_params))
            else:
                label_syn = label_net(image_syn.detach().to(args.device), flat_param=label_params)
            label_syn = torch.cat(batch_labels, dim=0)
            label_syn = torch.nn.functional.softmax(label_syn)
            del label_net, label_params
            for _ in batch_labels:
                del _

        syn_images = image_syn

        y_hat = label_syn.to(args.device)

        syn_image_gradients = torch.zeros(syn_images.shape).to(args.device)
        x_list = []
        original_x_list = []
        y_list = []
        indices_chunks = []
        gradient_sum = torch.zeros(student_params[-1].shape).to(args.device)
        indices_chunks_copy = []
        for _ in range(args.syn_steps):

            if not indices_chunks:
                indices = torch.randperm(len(syn_images))
                indices_chunks = list(torch.split(indices, args.batch_syn))

            these_indices = indices_chunks.pop()
            indices_chunks_copy.append(these_indices)

            x = syn_images[these_indices]
            this_y = y_hat[these_indices]
            # print(this_y)

            original_x_list.append(x)
            if args.dsa and (not args.no_aug):
                x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)
            x_list.append(x.clone())
            y_list.append(this_y.clone())

            forward_params = student_params[-1]
            x = student_net(x, flat_param=forward_params)
            ce_loss = criterion(x, this_y)

            grad = torch.autograd.grad(ce_loss, forward_params, create_graph=True, retain_graph=True)[0]

            detached_grad = grad.detach().clone()
            student_params.append(student_params[-1] - syn_lr.item() * detached_grad)
            gradient_sum += detached_grad

            del grad


        for i in range(args.syn_steps):

            w_i = student_params[i]
            output_i = student_net(x_list[i], flat_param = w_i)
            if args.batch_syn:
                ce_loss_i = criterion(output_i, y_list[i])
            else:
                ce_loss_i = criterion(output_i, y_hat)

            grad_i = torch.autograd.grad(ce_loss_i, w_i, create_graph=True, retain_graph=True)[0]
            single_term = syn_lr.item() * (target_params - starting_params)
            square_term = (syn_lr.item() ** 2) * gradient_sum
            gradients = 2  * torch.autograd.grad( (single_term + square_term) @ grad_i / param_dist, original_x_list[i])
            with torch.no_grad():
                syn_image_gradients[indices_chunks_copy[i]] += gradients[0]


        syn_images.grad = syn_image_gradients

        k = syn_image_gradients.flatten()
        wandb.log({"syn_image_gradients": k.dot(k)}, step=it)

        grand_loss = starting_params - syn_lr * gradient_sum - target_params

        grand_loss = grand_loss.dot(grand_loss) / param_dist

        lr_grad = torch.autograd.grad(grand_loss, syn_lr)[0]
        syn_lr.grad = lr_grad

        optimizer_img.step()
        optimizer_lr.step()
        

        if (it>=300) and (it%300 == 0):
            syn_save_for_exp1 = copy.deepcopy(image_syn.cuda())
            save_dir = save_path

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(syn_save_for_exp1.cpu(), os.path.join(save_dir, "images_{}.pt".format(it)))
            torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_{}.pt".format(it)))

        wandb.log({"Grand_Loss": grand_loss.detach().cpu(),
                   "Start_Epoch": start_epoch})

        for _ in student_params:
            del _
        if it%10 == 0:
            print('%s iter = %04d, loss = %.4f' % (get_time(), it, grand_loss.item()))
    e_txt_acc.close()
    end_time = time.time()
    
    execution_time = end_time - start_time
    max_acc = max(acc_list)
    txt_dir = './txt/'
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)

    file_name = "distill_efficiency_a5000_ipc1_10_50_cifar100.txt"
    e_txt = open(txt_dir+file_name, 'a')
    e_txt.write("attacker {0} distill time:{1}    {2}    {3}    {4}".format(args.attacker_num, execution_time,args.model,args.dataset,args.ipc)+"\n")
    e_txt.close()

    if args.dataset == 'CIFAR10':
        file_name_acc = "acc_cifar10.txt"
        acc_txt = open(txt_dir+file_name_acc, 'a')
        acc_txt.write("{0}    {1}    {2}    {3}    attacker_{4}".format(max_acc,args.model,args.dataset,args.ipc, args.attacker_num)+"\n")
        acc_txt.close()
    elif args.dataset == 'CIFAR100':
        file_name_acc = "acc_cifar100.txt"
        acc_txt = open(txt_dir+file_name_acc, 'a')
        acc_txt.write("{0}    {1}    {2}    {3}    attacker_{4}".format(max_acc,args.model,args.dataset,args.ipc, args.attacker_num)+"\n")
        acc_txt.close()
    print(f"time:{execution_time}s    {args.model} {args.dataset}")
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')

    parser.add_argument('--model', type=str, default='ConvNet', help='model')

    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')

    parser.add_argument('--num_eval', type=int, default=2, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')

    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=5000, help='how many distillation steps to perform')
    #
    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')

    parser.add_argument('--lr_init', type=float, default=0.01, help='how to init lr (alpha)')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    parser.add_argument('--pix_init', type=str, default='real', choices=["noise", "real"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')

    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')

    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')

    parser.add_argument('--data_path', type=str, default='./data', help='dataset path')
    parser.add_argument('--intermediate_result_save_path', type=str, default='/root/intermediate_result', help='buffer path')
    parser.add_argument('--syn_image_save_path', type=str, default='/root/intermediate_result/sym_image', help='image path')

    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=50, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')

    parser.add_argument('--zca', action='store_true',default=True, help="do ZCA whitening")
    parser.add_argument('--random_trajectory', action='store_true', default=False, help="using random trajectory instead of pretrained")

    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")

    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')

    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')

    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')
    parser.add_argument('--attacker_num', type=int, default=0, help='attacker number')

    parser.add_argument('--teacher_label', action='store_true', default=False, help='whether to use label from the expert model to guide the distillation process.')
    parser.add_argument('--times', type=int, default=0, help='number of times')
    args = parser.parse_args()

    main(args)


