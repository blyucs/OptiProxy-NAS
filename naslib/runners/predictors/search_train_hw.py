import logging
import torch
import os,sys

sys.path.append('.')
import numpy as np
import matplotlib.pyplot as plt
import glob
from efn_search_hw import Search_Model
from naslib import utils
import argparse
from naslib.utils import setup_logger,  create_cpfile_dir
from naslib.utils import setup_logger, get_dataset_api, create_cpfile_dir
from naslib.search_spaces.core.query_metrics import Metric


config, args = utils.get_config_from_args(config_type="st")
dataset_api = get_dataset_api(config.search_space, config.dataset, dataset_ver=config.dataset_ver)
torch.set_printoptions(threshold=5000, linewidth=200)
torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True)
utils.set_seed(config.seed)
logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)
utils.log_args(config)
scripts_to_save = glob.glob('naslib/runners/predictors/*.py')
scripts_to_save.append(args.config_file)
scripts_to_save.append('naslib/predictors/gcn.py')
scripts_to_save.append('naslib/utils/get_dataset_api.py')
scripts_to_save.append(f'{config.out_dir}/{config.dataset}/configs/predictors/config_gcn_search_train.yaml')
create_cpfile_dir(config.save, scripts_to_save=scripts_to_save)

if config.search_space == "nasbench101":
    max_acc = 0.0
    max_test_acc = 0.0
    max_str = ""
    max_test_str = ""
    sample_mean = False
    average_accuracies = []
    
    for item, value in dataset_api['nb101_data'].computed_statistics.items():
        if not config.sample_mean:
            sample_index = 0
            for dict_list in value.values():
                l = dict_list[sample_index]
                cur_acc = float(l['final_validation_accuracy'])
                if float(l['final_validation_accuracy']) > max_acc:
                    max_acc = cur_acc
                    max_str = item
                if cur_acc > 0.946:
                    logger.info(f'Good  acc = {cur_acc}, good str = {item}')
        else:
            
            all_accuracies = []
            test_all_accuracies = []
            
            for dict_list in value.values():
                
                accuracies = [float(l['final_validation_accuracy']) for l in dict_list if 'final_validation_accuracy' in l]
                all_accuracies.extend(accuracies)
                test_accuracies = [float(l['final_test_accuracy']) for l in dict_list if 'final_test_accuracy' in l]
                test_all_accuracies.extend(test_accuracies)
            
            if all_accuracies:
                average_accuracy = np.mean(all_accuracies)
                average_accuracies.append(average_accuracy)
                
                if average_accuracy > max_acc:
                    max_acc = average_accuracy
                    max_str = item
                
                if average_accuracy > 0.946:
                    logger.info(f'Good avg acc = {average_accuracy}, good str = {item}')

            if test_all_accuracies:
                test_average_accuracy = np.mean(test_all_accuracies)
                
                if test_average_accuracy > max_test_acc:
                    max_test_acc = test_average_accuracy
                    max_test_str = item

    
    logger.info(f'Global optimal average acc = {max_acc}, str = {max_str}')
    logger.info(f'Global optimal average test acc= {max_test_acc}, str = {max_test_str}')
    logger.info(f'mean:{ np.mean(average_accuracies)}, std: {np.std(average_accuracies)}')


if config.search_space == "nasbench301":
    l = []
    
    for item, value in dataset_api['nb301_data'].items():
        l.append(value['val_accuracies'][-1])
        if value['val_accuracies'][-1] > 94.5:
            print(f"item: {item}, acc: {value['val_accuracies'][-1]}")

    mean, std = np.mean(l), np.std(l)

    logger.info(f'mean = {mean}, std  ={std}')

if config.search_space == "nlp":
    l = []
    
    for item, value in dataset_api['nlp_data'].items():
        acc = 100 -  value['val_losses'][-1]
        l.append(acc)
        if acc > 95.5:
            print(f"item: {item}, acc: {acc}")

    mean, std = np.mean(l), np.std(l)

    logger.info(f'mean = {mean}, std  ={std}')


all_best_acc_arrays = []


utils.set_seed(config.seed)

pred_record = []
search_model = Search_Model(logger=logger, config=config, seed=config.seed, dataset_api=dataset_api, \
                            search_device = config.search_device, surrogate_device=config.surrogate_device)
search_model.sampling_and_query_init()
for step in range(config.search_para.search_steps):
    
    search_model.fit_surrogate_model(step_idx = step)
    pred_values = search_model.search(step_idx = step)
    search_model.sampling_and_query()
    logger.info(f"Seed {config.seed}, search step: {step}, best_acc: {search_model.best_acc}")
    pred_record = pred_record + pred_values

smooth_loss_values = []
alpha = 0.1 
running_loss = 0
for loss_value in pred_record:
    running_loss = alpha * loss_value + (1 - alpha) * running_loss
    smooth_loss_values.append(running_loss)
plt.figure(figsize=(10, 5))
plt.plot(pred_record, label='Original Loss', alpha=0.3)  
plt.plot(smooth_loss_values, label='Smoothed Loss', color='red')  
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Smoothed pred value Curve')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(config.save, f'pred_record_{config.seed}.png'))


all_best_acc_arrays.append(search_model.best_acc_array)

plt.figure(figsize=(10, 6))

plt.plot(search_model.steps, search_model.accuracies, marker='o', linestyle='-', color='b', label='Validation Accuracy')

plt.plot(search_model.steps, search_model.best_acc_array[1:], marker='x', linestyle='--', color='r', label='Best Validation Accuracy')

plt.ylim(config.ylim_s, config.ylim_e)
plt.title('Optimization Curve of Accuracy')
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.legend(loc="upper left")
plt.grid(True)
plt.savefig(os.path.join(config.save, f'accuracy_curve_{config.seed}.png'))





















