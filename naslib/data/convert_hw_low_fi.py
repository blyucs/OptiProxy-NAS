import torch
import pickle
import numpy as np
import sys
sys.path.append('.')
from naslib.search_spaces.nasbench201.conversions import *


ver_str = 'v1_0'
d = torch.load('./naslib/data/NAS-Bench-201-v1_0-e61699.pth')
objects = {}
max_acc = 0

sys.path.append('.')
dataset_list = ['cifar10', 'cifar100', 'ImageNet16-120']
for dataset in dataset_list:
    if dataset == 'cifar10':
        test_dataset_name = 'cifar10'
        dataset_name = 'cifar10-valid'
    elif dataset == 'cifar100':
        dataset_name = 'cifar100'
        test_dataset_name = 'cifar100'
    else:
        dataset_name = 'ImageNet16-120'
        test_dataset_name = 'ImageNet16-120'
    seed = 888
    vlist = []
    for i in range(len(d['arch2infos'])):
        try:
            cifar10 = {}
            cifar10['eval_acc1es'] = list(d['arch2infos'][i]['full']['all_results'][(dataset_name,seed)]['eval_acc1es'].values())
            mlist = []
            if (dataset_name,777) in d['arch2infos'][i]['full']['all_results'].keys():
                mlist.append(list(d['arch2infos'][i]['full']['all_results'][(dataset_name,777)]['eval_acc1es'].values())[-2])
            if (dataset_name,888) in d['arch2infos'][i]['full']['all_results'].keys():
                mlist.append(list(d['arch2infos'][i]['full']['all_results'][(dataset_name,888)]['eval_acc1es'].values())[-2])
            if (dataset_name,999) in d['arch2infos'][i]['full']['all_results'].keys():
                mlist.append(list(d['arch2infos'][i]['full']['all_results'][(dataset_name,999)]['eval_acc1es'].values())[-2])
            marray = np.array(mlist)
            m = marray.mean()
            cifar10['eval_acc1es'][-2] = m
            vlist.append(m)
            cost = {}
            cost['flops'] = d['arch2infos'][i]['full']['all_results'][(dataset_name,seed)]['flop']
            cost['params'] = d['arch2infos'][i]['full']['all_results'][(dataset_name,seed)]['params']
            cost['latency'] = d['arch2infos'][i]['full']['all_results'][(dataset_name,seed)]['latency'][-1]
            cost['train_time'] = float(0)
            cifar10['cost_info'] = cost
            objects[d['arch2infos'][i]['full']['arch_str']] = {dataset_name : cifar10}
        except:
            pass
    converted_objects = {}
    for k,v in objects.items():
        converted_objects[convert_str_to_op_indices(k)] = v

    objects={}
    max_val_acc = 0
    max_test_acc = 0
    vlist = []
    tlist = []
    for i in range(len(d['arch2infos'])):
        try:
            cifar10_valid = {}
            cifar10_valid['eval_acc1es'] = [0,0]

            #   valid
            mlist_val = []
            if (dataset_name,777) in d['arch2infos'][i]['full']['all_results'].keys():
                mlist_val.append(list(d['arch2infos'][i]['full']['all_results'][(dataset_name,777)]['eval_acc1es'].values())[-2])
            if (dataset_name,888) in d['arch2infos'][i]['full']['all_results'].keys():
                mlist_val.append(list(d['arch2infos'][i]['full']['all_results'][(dataset_name,888)]['eval_acc1es'].values())[-2])
            if (dataset_name,999) in d['arch2infos'][i]['full']['all_results'].keys():
                mlist_val.append(list(d['arch2infos'][i]['full']['all_results'][(dataset_name,999)]['eval_acc1es'].values())[-2])
            marray_val = np.array(mlist_val)
            m_val = marray_val.mean()
            cifar10_valid['eval_acc1es'][-2] = m_val
            vlist.append(m_val)

            if m_val > max_val_acc:
                max_val_acc = m_val
                max_val_index = i

            #  test
            mlist_test = []
            if (test_dataset_name,777) in d['arch2infos'][i]['full']['all_results'].keys():
                mlist_test.append(list(d['arch2infos'][i]['full']['all_results'][(test_dataset_name,777)]['eval_acc1es'].values())[-1])
            if (test_dataset_name,888) in d['arch2infos'][i]['full']['all_results'].keys():
                mlist_test.append(list(d['arch2infos'][i]['full']['all_results'][(test_dataset_name,888)]['eval_acc1es'].values())[-1])
            if (test_dataset_name,999) in d['arch2infos'][i]['full']['all_results'].keys():
                mlist_test.append(list(d['arch2infos'][i]['full']['all_results'][(test_dataset_name,999)]['eval_acc1es'].values())[-1])
            marray_test = np.array(mlist_test)
            m_test = marray_test.mean()
            cifar10_valid['eval_acc1es'][-1] = m_test
            tlist.append(m_test)

            if m_test > max_test_acc:
                max_test_acc = m_test
                max_test_index = i

            objects[d['arch2infos'][i]['full']['arch_str']] = {dataset_name : cifar10_valid}
        except:
            print("except")
            pass

    for k,v in objects.items():
        converted_objects[k] = v
    print(np.mean(vlist))
    print(np.std(vlist))
    print(np.max(vlist))
    filehandler = open("./naslib/data/nb201_"+dataset_name+"_"+f"val_test_{ver_str}.pickle", "wb")
    pickle.dump(converted_objects, filehandler)
    filehandler.close()
    print(np.max(vlist))
