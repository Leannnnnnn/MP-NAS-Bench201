# Note: 新版本的训练数据生成代码，考虑主干节点，Resblock节点位宽设置，各cell独立分配量化位宽
# Date: 2024/04/13

import os
import re
import glob
import yaml
import torch
import pickle
import random
import torchvision
from torch import nn
from tqdm import tqdm
from nats_bench import create
from utils.model_utils import load_data, train_model, test_model, train_model_with_epoch_list, get_network, find_nor_conv_positions
from utils.bitassign_utils import MixBitAssign




if __name__ == "__main__":
    # Prepare dataset
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=str, required=True, help='Model range, like 1-1000')
    parser.add_argument('--device', type=str, required=True, help='Device on which the models will be run, set 0 or 1')
    parser.add_argument('--total', type=int, default=50, help='The total number of model to train.')
    parser.add_argument('--cell_type', type=str, default='cell_group', help='The type of bitwidth assign in cell layers.', choices=['cell_group','cell_group_op_random','cell_uniform','cell_uniform_op_random','cell_separated','cell_separated_op_random']) # Raise error not in choice
    parser.add_argument('--stem_type', type=str, default='quant_8bit', help='The type of bitwidth assign in stem layers.', choices=['quant_8bit', 'quant_separated'])
    
    args = parser.parse_args()

    split_index = args.index.split('-')

    # 超参数设置
    H0 = {'dataset': 'cifar10','epochs': 200, 'lr': 0.1, 'batch_size': 256}
    H1 = {'dataset': 'cifar10','epochs': 200, 'lr': 0.05, 'batch_size': 256}
    H2 = {'dataset': 'cifar10','epochs': 150, 'lr': 0.01, 'batch_size': 256}
    H3 = {'dataset': 'cifar10','epochs': 150, 'lr': 0.01, 'batch_size': 1024}

    epoch_list = [50]

    total_model = args.total
    target_H = H2
    print(args)
    print('Train info: ',target_H)

    train_loader, valid_loader, test_loader = load_data('cifar10', '~/dataset', target_H)
    print('Dataset prepared.')

    api = create('/home/dell/dataset/NATS-tss-v1_0-3ffb9-simple', 'tss', fast_mode=True, verbose=False)

    yaml_path = '/home/dell/MP-NAS-Bench201/results/configs/'
    model_save_dir = '/home/dell/MP-NAS-Bench201/results/models'

    yaml_cache = {}

    cell_uniform_list_100_1 = [99, 289, 315, 631, 743, 806, 814, 1208, 1239, 1572, 1680, 1817, 1821, 2071, 2679, 2856, 2858, 2912, 2931, 
                             2986, 3461, 3521, 3800, 3826, 4139, 4156, 4166, 4367, 4472, 4811, 4950, 5009, 5468, 5823, 5883, 5896, 5922, 
                             6098, 6200, 6267, 6431, 6774, 6779, 6940, 7230, 7378, 7507, 7866, 8240, 8703, 8710, 8789, 8815, 8987, 9461, 
                             9667, 9694, 9773, 9791, 9964, 9971, 10095, 10113, 10155, 10171, 10296, 10695, 10885, 10937, 10992, 11403, 
                             11551, 11794, 11866, 12047, 12138, 12175, 12348, 12540, 12552, 12667, 12713, 12738, 12847, 12932, 12963, 12967, 
                             13121, 13186, 13502, 13788, 14230, 14252, 14337, 14878, 14928, 14972, 15002, 15121, 15579]

    cell_uniform_list_100_2 = [683, 815, 871, 1003, 1085, 1097, 1158, 1479, 1524, 1590, 1674, 1838, 1921, 2112, 2340, 2484, 2566, 2961, 
                               2974, 3067, 3224, 3277, 3477, 3567, 4064, 4168, 4202, 4524, 4576, 5146, 5165, 5226, 5376, 5555, 5753, 5809, 
                               5895, 6057, 6130, 6139, 6437, 6574, 6742, 6751, 6770, 7243, 7677, 7729, 7749, 7817, 8191, 8350, 8638, 8791, 
                               9036, 9104, 9150, 9483, 9580, 9706, 9805, 10217, 10327, 10458, 10597, 10642, 10734, 10865, 10972, 11122, 11141, 
                               11287, 11514, 11603, 11631, 11861, 11994, 12164, 12315, 12529, 12593, 12908, 12947, 13023, 13109, 13137, 13204, 
                               13645, 13661, 13879, 13924, 13929, 14367, 14639, 14850, 14951, 15141, 15347, 15418, 15624]

    cell_uniform_list_100_3 = [45, 64, 151, 161, 307, 326, 528, 571, 597, 864, 1014, 1250, 1634, 1668, 2090, 2297, 2523, 2632, 2718, 2735, 
                               2786, 2827, 2909, 2921, 3303, 3519, 4084, 4450, 4468, 4543, 4559, 5038, 5145, 5482, 5568, 5805, 5840, 5892, 
                               6289, 6383, 6634, 6669, 6735, 6790, 6953, 6982, 7111, 7256, 8034, 8379, 8713, 8807, 8853, 8921, 9039, 9120, 
                               9784, 9896, 9977, 10014, 10186, 10258, 10310, 10678, 10768, 10784, 11165, 11325, 11471, 11720, 11732, 11748, 
                               11766, 11926, 11996, 12029, 12030, 12057, 12199, 12206, 12306, 12327, 12338, 12578, 12731, 12772, 13024, 13133, 
                               13173, 13539, 13742, 13778, 14158, 14216, 14443, 14624, 14678, 14733, 14918, 15015]
    
    cell_uniform_list_100_4 = [295, 371, 382, 746, 854, 1155, 1280, 1448, 1552, 1747, 1758, 1840, 2000, 2021, 2592, 2940, 2987, 3044, 3051,
                            3142, 3552, 3713, 3825, 3871, 3982, 3990, 4178, 4445, 4866, 4891, 4952, 5006, 5105, 5435, 5557, 5815, 5909, 
                            6116, 6194, 6269, 6294, 6314, 6335, 6425, 6994, 7190, 7263, 7265, 7349, 7443, 7444, 7881, 8015, 8069, 8204, 
                            8296, 8338, 8340, 8531, 8560, 8707, 8889, 9032, 9351, 9353, 9355, 9494, 9585, 9944, 10223, 10240, 10303, 10917, 
                            10981, 11060, 11195, 11388, 11391, 11607, 11749, 11838, 12088, 12139, 12617, 12670, 12704, 12897, 12979, 13868, 
                            13875, 13902, 14004, 14056, 14162, 14274, 14713, 15065, 15188, 15207, 15234]
    
    cell_uniform_list_100_5 = [31, 177, 360, 379, 395, 429, 562, 646, 678, 1169, 1422, 1472, 1656, 1803, 1929, 2426, 2459, 2685, 2723, 2904, 
                           3325, 3766, 3766, 4305, 4453, 4596, 4927, 5140, 5183, 5205, 5283, 5309, 5488, 5931, 6158, 6207, 6291, 6570, 
                           6649, 6688, 6716, 6749, 6818, 6997, 7013, 7178, 7210, 7484, 7966, 8073, 8279, 8674, 9057, 9319, 9323, 9368, 
                           9677, 9855, 10087, 10114, 10285, 10446, 10615, 10774, 10814, 10966, 11279, 11314, 11477, 11661, 11671, 11678, 
                           11734, 11813, 12018, 12097, 12235, 12241, 12257, 12489, 12742, 12824, 12876, 13403, 13462, 13646, 13664, 13758, 
                           13863, 14280, 14280, 14435, 14437, 14722, 14913, 15343, 15449, 15558, 15560, 15605]
    
    cell_uniform_list_100_6 = [227, 257, 436, 473, 508, 725, 765, 766, 810, 840, 1187, 1312, 1388, 1617, 1713, 1810, 1883, 1946, 1953, 2074, 
                           2086, 2367, 2643, 2715, 3099, 3137, 3252, 3317, 3407, 3453, 3641, 3935, 3978, 4103, 4165, 4527, 4703, 4823, 4938, 
                           4976, 5021, 5396, 5454, 5467, 5501, 5800, 6076, 6209, 6246, 6315, 6432, 6721, 6863, 6875, 6965, 7054, 7390, 7858, 
                           8085, 8216, 8234, 8309, 8773, 9195, 9225, 9563, 9644, 9661, 9810, 9955, 9990, 10089, 10141, 10232, 10319, 10431, 10432, 
                           10653, 11773, 11815, 11958, 12079, 12503, 12968, 13163, 13385, 13692, 13718, 13886, 13913, 13913, 14003, 14057, 14622, 
                           14735, 14883, 14917, 15068, 15157, 15224]
    
    index_list_lower_0_6_212 = [151, 177, 315, 436, 528, 631, 646, 678, 683, 746, 765, 806, 864, 1003, 1014, 1097, 1155, 1158, 1208, 1239, 1312, 1422,
                             1448, 1552, 1634, 1747, 1758, 1817, 1838, 1929, 2000, 2021, 2071, 2112, 2340, 2426, 2459, 2592, 2643, 2679, 2715, 2723, 
                             2735, 2858, 2931, 2961, 2987, 3044, 3051, 3067, 3099, 3142, 3224, 3303, 3325, 3453, 3519, 3552, 3713, 3766, 3800, 4084, 
                             4103, 4166, 4168, 4472, 4524, 4576, 4596, 4703, 4811, 4823, 4891, 4950, 5006, 5145, 5146, 5205, 5309, 5435, 5501, 5753, 
                             5800, 5809, 5909, 6076, 6098, 6139, 6194, 6314, 6335, 6383, 6437, 6669, 6688, 6742, 6749, 6751, 6770, 6774, 6940, 7230, 
                             7265, 7378, 7484, 7677, 7817, 7858, 7866, 7881, 8015, 8069, 8073, 8191, 8240, 8338, 8340, 8350, 8379, 8674, 8710, 8713, 
                             8789, 8815, 8889, 8921, 9039, 9057, 9150, 9319, 9323, 9355, 9368, 9461, 9563, 9585, 9644, 9667, 9694, 9773, 9944, 9964, 
                             9971, 10171, 10223, 10258, 10296, 10303, 10310, 10327, 10432, 10446, 10458, 10597, 10642, 10653, 10734, 10768, 10784, 10814, 
                             10865, 10972, 10981, 11122, 11141, 11314, 11325, 11403, 11477, 11603, 11607, 11678, 11732, 11766, 11794, 11813, 11838, 12030, 
                             12057, 12079, 12097, 12199, 12241, 12315, 12667, 12713, 12738, 12772, 12847, 12908, 12932, 12968, 12979, 13173, 13186, 13462, 
                             13661, 13664, 13742, 13758, 13788, 13886, 14252, 14280, 14437, 14722, 14913, 15015, 15068, 15207, 15234, 15579, 15624]

    choosen_index_list = index_list_lower_0_6_212[int(split_index[0]): int(split_index[1])+1]

    def get_cache():
        cache = {}

        # 使用glob找出所有匹配的文件
        pattern = os.path.join(yaml_path, '*_*.yaml')
        files = glob.glob(pattern)
        for file_path in files:
            match = re.search(r'([0-9]+_[0-9]+).yaml', file_path)
            if match:
                dict_idx = match.group(1)
                split = dict_idx.split('_')
                model_idx = int(split[0])
                filename = yaml_path+'{}.yaml'.format(dict_idx)
                if model_idx in choosen_index_list: 
                    with open(filename, 'r') as file:
                        table = yaml.safe_load(file)
                    cache[table['index']] = table
        return cache
        

    for try_get_model in range(args.total):
        print(args)
        yaml_cache = get_cache()
        print('Info: {} caches loaded.'.format(len(yaml_cache)))
        print('Info: Try to get model {}/{} times. '.format(try_get_model+1, total_model))
        index = random.randint(0, int(split_index[1])-int(split_index[0]))
        model_idx = choosen_index_list[index]
        # model_idx = 13539

        # 保证同一个模型仅训练一次
        # if model_idx in get_meta_key(pkl_path):
        #     print('Model {} is collected, regenerate.'.format(model_idx))
        #     continue

        info = api.query_by_index(model_idx, hp = 200)

        cell_arch_str = info.arch_str
        print('Arch: {}'.format(cell_arch_str))
        conv_positions = find_nor_conv_positions(cell_arch_str)

        model = get_network(api, model_idx, dataset = target_H['dataset'], quant = True)
        bit_assigner = MixBitAssign(model, model_idx, target_H, conv_positions, cell_arch_str, yaml_cache, yaml_path = yaml_path)
        epoch_trained = {}

        epoch_trained = bit_assigner.generate_random_bitwidth(cell_type = args.cell_type, stem_type = args.stem_type)  

        not_train = {}
        not_train_epoch_list = []
        
        dict_split = bit_assigner.get_dict_index().split('_')

        num = int(dict_split[1])
        for e in epoch_list:
            if e not in epoch_trained.keys():
                not_train[e] = dict_split[0] + '_' + str(num)
                not_train_epoch_list.append(e)
                num += 1
        not_train_epoch_list.sort()
        print('Set bit width: {}'.format(bit_assigner.config_table['bit_width'])) # 确保没有生成重复的位宽选项

        model = bit_assigner.get_model()

        epoch_result = train_model_with_epoch_list(model, target_H, model_save_dir, train_loader, valid_loader, test_loader, device = args.device, epoch_list = epoch_list, not_train = not_train, epoch_trained = epoch_trained, not_train_epoch_list = not_train_epoch_list)
        for e in epoch_result.keys():
            bit_assigner.save_to_yaml(epoch_result[e]['dict_name'], e, epoch_result[e]['val_acc'], epoch_result[e]['val_loss'], epoch_result[e]['test_acc'], epoch_result[e]['test_loss'])
            yaml_cache[epoch_result[e]['dict_name']] = bit_assigner.get_yaml_info()

        '''
        if len(not_train) > 0:
            print('Wait to train: ', not_train)
            for i, e in enumerate(epoch_list):
                if e in not_train.keys():
                    if i != 0: 
                        if epoch_list[i-1] not in not_train.keys():
                            load = (epoch_trained[epoch_list[i-1]], epoch_list[i-1])
                        else:
                            load = None
                    else:
                        load = None
                    target_H['epochs'] = e
                else:
                    continue

                print('Try to get model {}/{} times of index: {}'.format(try_get_model+1, total_model, not_train[e]))
                print(args)
                train_model(model, target_H, model_save_dir, train_loader, not_train[e], device = args.device, load = load)

                print('Validate model {}.'.format(not_train[e]))
                val_acc, val_loss = test_model(model, valid_loader, len(valid_loader)*target_H['batch_size'])

                print('Test model {}.'.format(not_train[e]))
                test_acc, test_loss = test_model(model, test_loader, len(test_loader)*target_H['batch_size'])
                
                bit_assigner.save_to_yaml(not_train[e], target_H['epochs'], val_acc, val_loss, test_acc, test_loss)
                yaml_cache[not_train[e]] = bit_assigner.get_yaml_info()

                epoch_trained[e] = not_train[e]
                del not_train[e]

        else:
            print('Exist all epoch choices, continue.')
            continue
        '''

