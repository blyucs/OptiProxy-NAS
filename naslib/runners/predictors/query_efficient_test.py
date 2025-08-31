import logging
import torch
import os,sys

import numpy as np
import matplotlib.pyplot as plt
import glob
from efn_search import Search_Model
from naslib import utils
import argparse
from naslib.utils import setup_logger,  create_cpfile_dir
from naslib.utils import setup_logger, get_dataset_api, create_cpfile_dir, create_exp_dir, get_project_root
import time
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")a

def get_config_from_args(config_file=None, config_type="nas"):
    """
    Parses command line arguments and merges them with the defaults
    from the config file.

    Prepares experiment directories.

    Args:
        args: args from a different argument parser than the default one.
    """

    if config_file is None:
        config = utils.load_default_config(config_type=config_type)
    else:
        config = utils.load_config(path=config_file)

    if config_type == "st":
        config.save = "{}/{}/{}/{}/{}/{}/{}".format(
            config.out_dir,
            config.dataset,
            "predictors",
            config.predictor,
            config.experiment_type,
            config.config_name, # +"_"+str(config.test_id),
            time.strftime("%Y%m%d-%H%M%S")
        )

    else:
        print("invalid config type in utils/utils.py")

    config.data = "{}/data".format(get_project_root())
    print(config.data)
    print("---------")

    create_exp_dir(config.save)
    # create_exp_dir(config.save + "/search")  # required for the checkpoints
    # create_exp_dir(config.save + "/eval")

    return config
def config_init(config_yaml = None):
    config = get_config_from_args(config_file=config_yaml, config_type="st")
    dataset_api = get_dataset_api(config.search_space, config.dataset, dataset_ver=config.dataset_ver)
    torch.set_printoptions(threshold=5000, linewidth=200)
    torch.set_printoptions(sci_mode=False)
    np.set_printoptions(suppress=True)
    # utils.set_seed(config.seed)
    logger = setup_logger(config.save + "/log.log")
    logger.setLevel(logging.INFO)
    utils.log_args(config)
    scripts_to_save = glob.glob('naslib/runners/predictors/*.py')
    # scripts_to_save.append(args.config_file)
    scripts_to_save.append('naslib/predictors/gcn.py')
    scripts_to_save.append('naslib/utils/get_dataset_api.py')
    scripts_to_save.append(f'{config.out_dir}/{config.dataset}/configs/predictors/config_gcn_search_train.yaml')
    # create_cpfile_dir(config.save, scripts_to_save=scripts_to_save)
    return config, dataset_api, logger

all_best_acc_arrays = []

def run_exp_nas(seed, config, dataset_api, logger):
    # for t in range(config.num_seed):
    utils.set_seed(seed)
    pred_record = []
    search_model = Search_Model(logger=logger, config=config, seed=seed, dataset_api=dataset_api, \
                                search_device = config.search_device, surrogate_device=config.surrogate_device)
    search_model.sampling_and_query_init()
    for step in range(config.search_para.search_steps):
        # search_model.surrogate_model_init()
        search_model.fit_surrogate_model(step_idx = step)
        pred_values = search_model.search(step_idx = step)
        search_model.sampling_and_query()
        logger.info(f"Seed {seed}, search step: {step}, searched best arch's acc: {search_model.best_acc}")
        pred_record = pred_record + pred_values
    return search_model.best_acc_array

# mean_acc = np.mean(all_best_acc_arrays, axis=0)
# std_dev_acc = np.std(all_best_acc_arrays, axis=0)
# logger.info(f"Cur seed {config.seed}:")
# logger.info(f"Mean Best Validation Accuracy (over trials): {mean_acc}")
# logger.info(f"Variance of Best Validation Accuracy (over trials): {std_dev_acc}")

# all_best_acc_arrays = np.array(all_best_acc_arrays)  # Convert list of lists to a numpy array if not already done
# steps = range(mean_acc.shape[0])
# plt.figure(figsize=(10, 6))
# plt.plot(steps, mean_acc, color='g', label='Mean Best Validation Accuracy')
# plt.fill_between(steps, mean_acc - std_dev_acc, mean_acc + std_dev_acc, color='g', alpha=0.2, label='Std Dev Range')
# plt.xlabel('Steps')
# plt.ylim(config.ylim_s, config.ylim_e)
# plt.ylabel('Best Validation Accuracy')
# plt.title('Mean and Std Dev of Best Validation Accuracy Across Trials')
# plt.legend()
# plt.savefig(os.path.join(config.save, f'mul_t_mean_std.png'))
# plt.show()
# plt.close()