import torch
import pickle
import numpy as np

###############SET THIS##############
ver_str = 'v1_0'
# ver_str = 'v1_1'
# dataset = 'cifar10'
# dataset = 'cifar100'
dataset = 'ImageNet16-120'
################SET THIS##############
if dataset == 'cifar10':
    test_dataset_name = 'cifar10'
    dataset_name = 'cifar10-valid'
elif dataset == 'cifar100':
    dataset_name = 'cifar100'
    test_dataset_name = 'cifar100'
else:
    dataset_name = 'ImageNet16-120'
    test_dataset_name = 'ImageNet16-120'

if ver_str == 'v1_1':
    d = torch.load('./naslib/data/NAS-Bench-201-v1_1-096897.pth') # 4G
elif ver_str == 'v1_0':
    d = torch.load('./naslib/data/NAS-Bench-201-v1_0-e61699.pth') # 2G
objects = {}
max_val_acc = 0
max_test_acc = 0

#######################CIFAR10 #############################    TO APPEND THE TEST DATA
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
print("mean of val", np.mean(vlist))
print("std of val", np.std(vlist))
filehandler = open("./nb201_"+dataset_name+"_"+f"val_test_{ver_str}.pickle", "wb")
pickle.dump(objects, filehandler)

print("max val acc", max_val_acc,":",max_val_index)
print("max test acc", max_test_acc,":",max_test_index)
#  max valid  corr test
print("max val corresponding test acc:", tlist[max_val_index])
# max test corr valid
print("max test corresponding val acc:", vlist[max_test_index])
