# The functions below setup and run the search experiments

def update_config(config, optimizer_type, search_space_type, dataset, seed):
    # Dataset being used
    config.dataset = dataset

    # Directory to which the results/logs will be saved
    config.save = f"runs/{optimizer_type.__name__}/{search_space_type.__name__}/{dataset}/{seed}"

    # Seed used during search phase of the optimizer
    config.search.seed = seed


def run_optimizer(optimizer_type, search_space_type, dataset, config, seed, logger):
    # Update the config
    update_config(config, optimizer_type, search_space_type, dataset, seed)

    # Make the results directories
    os.makedirs(config.save + '/search', exist_ok=True)
    os.makedirs(config.save + '/eval', exist_ok=True)

    # Set up the loggers
    # logger = setup_logger()
    # logger.setLevel(logging.INFO)

    # See the config
    logger.info(f'Configuration is \n{config}')

    # Set up the seed
    utils.set_seed(seed)
    # Instantiate the search space
    n_classes = {
        'cifar10': 10,
        'cifar100': 100,
        'ImageNet16-120': 120
    }
    if search_space_type == NasBench201SearchSpace:
        search_space = search_space_type(n_classes=n_classes[dataset])
    else:
        search_space = search_space_type()
    search_space.instantiate_model = False
    # Get the benchmark API
    logger.info('Loading Benchmark API')
    dataset_api = get_dataset_api(search_space.get_type(), dataset, dataset_ver='v1_0')  # dataset_ver only for 201, only_val for pickle

    # Instantiate the optimizer and adapat the search space to the optimizer
    optimizer = optimizer_type(config)
    optimizer.adapt_search_space(search_space, dataset_api=dataset_api)

    # Create a Trainer
    trainer = Trainer(optimizer, config)

    # Perform the search
    trainer.search(report_incumbent=False)

    # Get the results of the search
    search_trajectory = trainer.search_trajectory
    print('Train accuracies:', search_trajectory.train_acc)
    print('Validation accuracies:', search_trajectory.valid_acc)
    # Get the validation performance of the best model found in the search phase
    best_model_val_acc = trainer.evaluate(dataset_api=dataset_api, metric=Metric.VAL_ACCURACY)
    # best_model_val_acc

    best_model = optimizer.get_final_architecture()

    return search_trajectory, best_model, best_model_val_acc

import sys
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import FormatStrFormatter
from naslib.runners.predictors.query_efficient_test import run_exp_nas, config_init
import argparse
# plot performance vs runtime
from naslib.runners.predictors.plt_query_efficient import plot_optimizers
# didn't run them long enough to do logspace here. (These experiments took surprisingly long to run)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = 'dotted'
# set up colors and plot markings
defaults = [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
            (1.0, 0.4980392156862745, 0.054901960784313725),
            (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
            (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
            (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
            (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
            (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
            (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
            (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
            (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)]

# goes up to 24
c_max = 9
colors = [*defaults[:c_max], *defaults[:c_max], *defaults[:c_max]]
fmts = [*['-'] * c_max, *['--'] * c_max, *[':'] * c_max]
markers = [*['^'] * c_max, *['v'] * c_max, *['o'] * c_max]

def get_results(results, metric='valid_acc', dataset='cifar10', ug=False):
    output = []
    time = []
    for result in results:
        val_acc = result['valid_acc']
        surr_time = np.array(result['runtime'])
        if ug:
            runtime = 200 * np.array(result['train_time']) + surr_time
        else:
            runtime = np.array(result['train_time']) + surr_time
        # val_err = [100 - x for x in val_acc]
        val_err = val_acc
        val_incumbent = [
            # min(val_err[:epoch]) for epoch in range(1,
            #                                         len(val_err) + 1)
            max(val_err[:epoch]) for epoch in range(1,
                                                    len(val_err) + 1)
        ]
        runtime = [
            sum(runtime[:epoch]) for epoch in range(1,
                                                    len(runtime) + 1)
        ]
        if metric == 'valid_acc':
            incumbent = val_incumbent
        elif metric == 'test_acc':
            test_err = [100 - x for x in result['test_acc']]
            inc_idx, best, best_idx = [], np.inf, 0
            for i, err in enumerate(val_err):
                if err < best:
                    best, best_idx = err, i
                    inc_idx.append(best_idx)
                    incumbent = [test_err[idx] for idx in inc_idx]
        output.append(incumbent)
        time.append(runtime)
    output = np.array(output)
    time = np.array(runtime)
    mean = np.mean(output, axis=0)
    std = np.std(output, axis=0)
    std_error = np.sqrt(
        np.var(output, axis=0, ddof=1) / np.asarray(output).shape[0])
    return mean, std, std_error, time

#initialize trajectories for RS

#imports we will use throught this section
import json
import logging
import os

# import the Trainer used to run the optimizer on a given search space
from naslib.defaults.trainer import Trainer
# import the optimizers
from naslib.optimizers import (
    RandomSearch,
    RegularizedEvolution,
    Bananas,
    Npenas,
    LocalSearch
)
# import the search spaces
from naslib.search_spaces import (
    NasBench101SearchSpace,
    NasBench201SearchSpace,
    NasBench301SearchSpace,
    NasBenchNLPSearchSpace
)

from naslib.search_spaces.core.query_metrics import Metric
from naslib import utils
from naslib.utils import get_dataset_api
from naslib.utils.log import setup_logger

from fvcore.common.config import CfgNode

# Set the optimizer and search space types
# They will be instantiated inside run_optimizer
# optimizer_type = RandomSearch  # {RegularizedEvolution, RandomSearch}
# search_space = 'nasbench201'
# Set the dataset
# dataset = 'cifar10'  # cifar10 for NB101 and NB301, {cifar100, ImageNet16-120} for NB201

supported_search_spaces = {
    "nasbench101": NasBench101SearchSpace,
    "nasbench201": NasBench201SearchSpace,
    "nasbench301": NasBench301SearchSpace,
    "nlp": NasBenchNLPSearchSpace,
    # 'transbench101_micro': TransBench101SearchSpaceMicro(config.dataset),
    # 'transbench101_macro': TransBench101SearchSpaceMacro(),
    # "asr": NasBenchASRSearchSpace(),
}

fidelity_epoch = {
    "nasbench101": -1,
    "nasbench201": -2,
    "nasbench301": -1,
    "nlp": -1
}

query_epochs = {
    "nasbench101":
        {'cifar10': 150},
    "nasbench201":
        {
            'cifar10': 100,
            'cifar100': 100,
            'ImageNet16-120': 280,
        },
    "nasbench301":
        {
            'cifar10': 100,
        },
    "nlp":
        {
            'ptb': 150
        }
}

def main():
    parser = argparse.ArgumentParser(description="Save trajectories merge to a file.")
    parser.add_argument('--search_space', type=str,  default='nasbench101', help="Search space name")
    # parser.add_argument('--dataset', type=str,  default='ImageNet16-120', help="Dataset name")
    parser.add_argument('--dataset', type=str,  default='cifar10', help="Dataset name")
    args = parser.parse_args()

    search_space = args.search_space
    dataset = args.dataset

    search_space_type = supported_search_spaces[
        search_space]  # NasBench201SearchSpace  # {NasBench101SearchSpace, NasBench201SearchSpace, NasBench301SearchSpace}

    yaml_config = {
        "nasbench101":
            {
                "cifar10":
                    {
                        'file': "p101-0/cifar10/configs/predictors/config_gcn_search_train.yaml",
                        'exp_name': 'NAS-Bench-101'
                    }
            },
        "nasbench201":
            {
                "cifar10":
                    {
                        'file': "p201-0/cifar10/configs/predictors/config_gcn_search_train.yaml",
                        'exp_name': 'NAS-Bench-201 - CIFAR10'
                    },
                "cifar100":
                    {
                        'file': "p201-0/cifar100/configs/predictors/config_gcn_search_train.yaml",
                        'exp_name': 'NAS-Bench-201 - CIFAR100'
                    },
                "ImageNet16-120":
                    {
                        'file': "p201-0/ImageNet16-120/configs/predictors/config_gcn_search_train_280.yaml",
                        'exp_name': 'NAS-Bench-201 - ImageNet16-120'
                    }
            },
        "nasbench301":
            {
                "cifar10":
                    {
                        'file': "p301-0/cifar10/configs/predictors/config_gcn_search_train_100.yaml",
                        'exp_name': 'NAS-Bench-301'
                    }
            },
        "nlp":
            {
                "ptb":
                    {
                        'file': "pnlp-0/ptb/configs/predictors/config_gcn_search_train_150.yaml",
                        'exp_name': 'NAS-Bench-NLP'
                    }
            }

    }

    seeds = range(10)
    # seeds = [1]
    optimizer_configs = {
        "Random Search": {
            'run': True,
            # 'run': False,
            'optimizer_type': RandomSearch,
            'config': {
                'search': {
                    # Required by Trainer
                    'epochs': query_epochs[search_space][dataset],
                    'checkpoint_freq': 200,

                    # Required by Random Search optimizer
                    'fidelity': fidelity_epoch[search_space],
                }
            }
        },

        "Local Search": {
            'run': True,
            # 'run': False,
            'optimizer_type': LocalSearch,
            'config': {
                'search': {
                    'epochs': query_epochs[search_space][dataset],
                    'checkpoint_freq': 100,
                    'fidelity': fidelity_epoch[search_space],
                    # 'k': 20,
                    "num_init": 20,
                    # "num_ensemble": 1,
                    # 'sample_size': 10,
                    # 'population_size': 30,
                    # "predictor_type": "mlp",
                    # "acq_fn_type": "its",
                    # "acq_fn_optimization": 'random_sampling',
                    # "encoding_type": 'adjacency_one_hot',
                    # "num_arches_to_mutate": 5,
                    # "max_mutations": 1,
                    # "num_candidates": 200,
                    # "batch_size": 256,
                    # "data_size": 25000,
                    # "cutout": False,
                    # "cutout_length": 16,
                    # "cutout_prob": 1.0,
                    # "train_portion": 0.7  # delete ?
                }
            }
        },

        "Regularized Evolution": {
            'run': True,
            # 'run': False,
            'optimizer_type': RegularizedEvolution,
            'config': {
                'search': {
                    # Required by Trainer
                    'epochs': query_epochs[search_space][dataset],
                    'checkpoint_freq': 100,

                    # Required by Random Search optimizer
                    'fidelity': fidelity_epoch[search_space],

                    # Required by RegularizedEvolution
                    'sample_size': 10,
                    'population_size': 30,
                }
            }
        },

        "BANANAS": {
            'run': True if search_space != 'nlp' else False,
            # 'run': False,
            'optimizer_type': Bananas,
            'config': {
                'search': {
                    # Required by Trainer
                    'epochs': query_epochs[search_space][dataset],
                    'checkpoint_freq': 100,

                    # Required by Random Search optimizer
                    'fidelity': fidelity_epoch[search_space],
                    'k': 20,
                    "num_init": 10,
                    "num_ensemble": 1,
                    # Required by RegularizedEvolution
                    'sample_size': 10,
                    'population_size': 30,
                    "predictor_type": 'bananas',
                    "acq_fn_type": "its",
                    "acq_fn_optimization": 'random_sampling',
                    "encoding_type": 'path',
                    "num_arches_to_mutate": 5,
                    "max_mutations": 1,
                    "num_candidates": 100
                }
            }
        },

        "NPENAS-BO": {
            'run': True,
            # 'run': False,
            'optimizer_type': Npenas,
            'config': {
                'search': {
                    'epochs': query_epochs[search_space][dataset],
                    'checkpoint_freq': 100,
                    'fidelity': fidelity_epoch[search_space],
                    'k': 20,
                    "num_init": 20,
                    "num_ensemble": 1,
                    'sample_size': 10,
                    'population_size': 30,
                    "predictor_type": "gp",
                    "acq_fn_type": "its",
                    "acq_fn_optimization": 'its',
                    "encoding_type": 'adjacency_one_hot',
                    "num_arches_to_mutate": 5,
                    "max_mutations": 1,
                    "num_candidates": 200,
                    "batch_size": 256,
                    "data_size": 25000,
                    "cutout": False,
                    "cutout_length": 16,
                    "cutout_prob": 1.0,
                    # "train_portion": 0.7  # delete ?
                }
            }
        },

        "NPENAS-NP": {
            'run': True,
            # 'run': False,
            'optimizer_type': Npenas,
            'config': {
                'search': {
                    'epochs': query_epochs[search_space][dataset],
                    'checkpoint_freq': 100,
                    'fidelity': fidelity_epoch[search_space],
                    'k': 20,
                    "num_init": 20,
                    "num_ensemble": 1,
                    'sample_size': 10,
                    'population_size': 30,
                    "predictor_type": "gcn",
                    "acq_fn_type": "its",
                    "acq_fn_optimization": 'its',
                    "encoding_type": 'adjacency_one_hot',
                    "num_arches_to_mutate": 5,
                    "max_mutations": 1,
                    "num_candidates": 200,
                    "batch_size": 256,
                    "data_size": 25000,
                    "cutout": False,
                    "cutout_length": 16,
                    "cutout_prob": 1.0,
                    # "train_portion": 0.7  # delete ?
                }
            }
        },

        "SurGrad-NAS(Ours)":
            {
                'run': True,
                # 'run': False,
                'yaml': yaml_config[search_space][dataset]
            }

    }

    trajectories_merge = {optimizer_key: [] for optimizer_key, optimizer_info in optimizer_configs.items() if optimizer_info['run']}
    config, dataset_api, logger = config_init(yaml_config[search_space][dataset]['file'])
    for optimizer_key, optimizer_info in optimizer_configs.items():
        if optimizer_key == 'SurGrad-NAS(Ours)':
            if optimizer_info['run']:
                for seed in seeds:
                    search_trajectory = run_exp_nas(seed, config, dataset_api, logger)
                    trajectories_merge[optimizer_key].append(search_trajectory)
        elif optimizer_info['run']:
            optimizer_type = optimizer_info['optimizer_type']
            config_values = optimizer_info['config']
            config_run = CfgNode.load_cfg(json.dumps(config_values))
            for seed in seeds:
                search_trajectory, best_model, best_model_val_acc = run_optimizer(
                    optimizer_type,
                    search_space_type,
                    dataset,
                    config_run,
                    seed,
                    logger
                )
                trajectories_merge[optimizer_key].append(search_trajectory)

    # multi threads
    # import concurrent.futures
    # def run_optimizer_parallel(optimizer_key, optimizer_info, seeds, config, dataset_api, logger, trajectories_merge):
    #     if optimizer_key == 'SurGrad-NAS(Ours)':
    #         if optimizer_info['run']:
    #             with concurrent.futures.ThreadPoolExecutor() as executor:
    #                 futures = [executor.submit(run_exp_nas, seed, config, dataset_api, logger) for seed in seeds]
    #                 for future in concurrent.futures.as_completed(futures):
    #                     search_trajectory = future.result()
    #                     trajectories_merge[optimizer_key].append(search_trajectory)
    #     elif optimizer_info['run']:
    #         optimizer_type = optimizer_info['optimizer_type']
    #         config_values = optimizer_info['config']
    #         config_run = CfgNode.load_cfg(json.dumps(config_values))
    #         with concurrent.futures.ThreadPoolExecutor() as executor:
    #             futures = [
    #                 executor.submit(run_optimizer, optimizer_type, search_space_type, dataset, config_run, seed, logger)
    #                 for seed in seeds]
    #             for future in concurrent.futures.as_completed(futures):
    #                 search_trajectory, best_model, best_model_val_acc = future.result()
    #                 trajectories_merge[optimizer_key].append(search_trajectory)
    #
    # for optimizer_key, optimizer_info in optimizer_configs.items():
    #     run_optimizer_parallel(optimizer_key, optimizer_info, seeds, config, dataset_api, logger, trajectories_merge)

    import pickle
    file_path = os.path.join(config.save, f"trajectories_merge_{yaml_config[search_space][dataset]['exp_name']}.pkl")
    with open(file_path, 'wb') as f:
        pickle.dump(trajectories_merge, f)
    # Plot RS for three seeds
    # plot_optimizers(trajectories_merge,"NAS-Bench-201 cifar10", x_runtime=True)
    plot_optimizers(config.search_space, config.dataset, trajectories_merge, yaml_config[search_space][dataset]['exp_name'], optimum_value=config.max_acc, x_runtime=False, save = config.save)

if __name__ == "__main__":
    main()
