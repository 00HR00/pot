import torch
import math
import torch.nn as nn
from scipy.stats import wasserstein_distance

layer_num = 0

def max_w_layer_EMD(model1, model2):
    l1_norm = 0
    l2_norm = 0
    max_value = -math.inf
    num_elements = 0
    average = 0

    model1.cpu()
    model2.cpu()

    for (key1, weight1), (key2, weight2) in zip(model1.state_dict().items(), model2.state_dict().items()):
        if key1 != key2:
            raise ValueError(f"Model layers do not match: {key1} and {key2}")

        diff = weight1 - weight2
        flat_w1 = torch.flatten(weight1)
        flat_w2 = torch.flatten(weight2)
        temp = wasserstein_distance(flat_w1, flat_w2)
        if temp > max_value:
            max_value = temp
    return max_value

#region
def max_w_layer_EMD_attacker(model1, model2, max_h):
    l1_norm = 0
    l2_norm = 0
    max_value = -math.inf
    num_elements = 0
    average = 0
    num = 0

    model1.cpu()
    model2.cpu()
    layer_num = 0
    for (key1, weight1), (key2, weight2) in zip(model1.state_dict().items(), model2.state_dict().items()):
        if key1 != key2:
            raise ValueError(f"Model layers do not match: {key1} and {key2}")

        diff = weight1 - weight2
        layer_num = layer_num + 1
        flat_w1 = torch.flatten(weight1)
        flat_w2 = torch.flatten(weight2)

        temp = wasserstein_distance(flat_w1, flat_w2)
        if temp > max_h:
            num = num + 1
        if temp > max_value:
            max_value = temp

    return max_value,num,layer_num
#endregion

def find_max_w():

    for k in range(1,77):
        max_value_between_model = -math.inf
        max_value_between_model_h = -math.inf

        #region
        nn = 0
        layer_nn = 0

        for i in range(0,150,k):
            if i<(150-k):

                model_i_h = torch.load('/root/intermediate_result/emd_honest/teacher_net_{}.pt'.format(i))
                model_i_plus_1_h = torch.load('/root/intermediate_result/emd_honest/teacher_net_{}.pt'.format(i+k))

                max_value_layer_h = max_w_layer_EMD(model_i_h,model_i_plus_1_h)

                if max_value_layer_h > max_value_between_model_h:
                    max_value_between_model_h = max_value_layer_h


        for i in range(0,150,k):
            n=0
            if i<(150-k):

                model_i = torch.load('/root/intermediate_result/attacker_5_cifar10_ConvNetD2_t/teacher_net_{}.pt'.format(i))
                model_i_plus_1 = torch.load('/root/intermediate_result/attacker_5_cifar10_ConvNetD2_t/teacher_net_{}.pt'.format(i+k))

                max_value_layer,n,layer_n = max_w_layer_EMD_attacker(model_i,model_i_plus_1,max_value_between_model_h)

                if max_value_layer > max_value_between_model:
                    max_value_between_model = max_value_layer

            nn = nn + n
            layer_nn = layer_nn + layer_n
        print(k)
        print("honest:{0}".format(max_value_between_model_h))
        print("attacker:{0}".format(max_value_between_model))
        print("layer num:{0}".format(layer_nn))
        print("greater than honest num:{0}".format(nn))
        print("\n")
        #endregion
    
find_max_w()