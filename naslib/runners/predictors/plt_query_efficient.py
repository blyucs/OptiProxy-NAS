import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from naslib import utils
import time

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
                    'file': "p301-0/cifar10/configs/predictors/config_gcn_search_train.yaml",
                    'exp_name': 'NAS-Bench-301'
                }
        },
    "nlp":
        {
            "ptb":
                {
                    'file': "pnlp-0/ptb/configs/predictors/config_gcn_search_train.yaml",
                    'exp_name': 'NAS-Bench-NLP'
                }
        }

}
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

def plot_optimizers(search_space, dataset, trajectories, benchmark, optimum_value, x_runtime=False, save = None):
    plot_zoomed = False
    plot_sem = True
    fig, ax = plt.subplots(figsize=[10, 8], dpi=600)
    if plot_zoomed:
        sub_axes = plt.axes([.6, .6, .25, .25])
    i = 0
    for optimizer in trajectories.keys():
        if optimizer == 'SurGrad-NAS(Ours)':
            mean = np.mean(trajectories['SurGrad-NAS(Ours)'], axis=0)
            std = np.std(trajectories['SurGrad-NAS(Ours)'], axis=0)
            std_error = np.sqrt(np.var(trajectories['SurGrad-NAS(Ours)'], axis=0, ddof=1) / np.asarray(trajectories['SurGrad-NAS(Ours)']).shape[0])
            time = None
            results_dict = {
                'label': 'OptiProxy-NAS (Ours)',
                'mean': mean,
                'runtime': time,
                'std_error': std_error,
                'std': std
            }
        else:
            mean, std, std_error, time = get_results(trajectories[optimizer])
            results_dict = {
                'label': optimizer,
                'mean': mean,
                'runtime': time,
                'std_error': std_error,
                'std': std
            }
        mean = results_dict['mean']
        if x_runtime:
            x = results_dict['runtime']
        else:
            x = range(len(mean))
        sem = results_dict['std_error'] # use std error
        # sem = results_dict['std']  # use std
        label = results_dict["label"]
        ax.plot(x, mean, label=label, color=colors[i], linestyle=fmts[i], linewidth = 2)
        ax.fill_between(x, mean - 1 * sem, mean + 1 * sem, color=colors[i], alpha=0.2)
        i += 1

    if search_space != 'nasbench301' and search_space != 'nlp':
        ax.axhline(y=optimum_value, color='blue', linestyle='--', linewidth=2, alpha=0.4)

        ax.annotate(f'Optimum*({optimum_value:.2f})', xy=(0, optimum_value), xytext=(-20, -12),
                    textcoords='offset points', ha='left', va='center', color='blue', fontsize=20)

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    if x_runtime:
        ax.set_xscale('log')
        ax.set_xlabel('Runtime [s] (simulated)', fontsize = 20)
    else:
        ax.set_xlabel('Number of Queries/Evaluations', fontsize = 25)
    ax.tick_params(axis='both', which='major', labelsize=22)

    ax.legend(loc=(0.45, 0.05), fontsize = 22)
    ax.set_ylabel('Val Acc. (%)', fontsize = 25)
    ax.grid(True, which="both", ls="-", alpha=.5)
    ax.set_title(benchmark, fontsize = 30)
    if search_space == 'nasbench201' and dataset == 'ImageNet16-120':
        ax.set_ylim(40, 47)
    fig.tight_layout()
    plt.savefig(os.path.join(save, f'query_efficient-{search_space}-{dataset}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save, f'query_efficient-{search_space}-{dataset}.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == "__main__":
    # 101
    # search_space = 'nasbench101'
    # dataset = 'cifar10'  # cifar10 for NB101 and NB301, {cifar100, ImageNet16-120} for NB201
    # file_path = "./p101-0/cifar10/predictors/gcn/search/test_101/20240702-223735/trajectories_merge_NAS-Bench-101.pkl"  # Replace <exp_name> with the actual experiment name
    # 201
    # search_space = 'nasbench201'
    # dataset = 'cifar10'  # cifar10 for NB101 and NB301, {cifar100, ImageNet16-120} for NB201
    # dataset = 'cifar100'  # cifar10 for NB101 and NB301, {cifar100, ImageNet16-120} for NB201
    # dataset = 'ImageNet16-120'  # cifar10 for NB101 and NB301, {cifar100, ImageNet16-120} for NB201
    # file_path = "./p201-0/cifar10/predictors/gcn/search/acc_0516_c10/20240703-112840/trajectories_merge_NAS-Bench-201 - CIFAR10.pkl"  # Replace <exp_name> with the actual experiment name
    # file_path = "./p201-0/cifar100/predictors/gcn/search/acc_0516_c10/20240703-112855/trajectories_merge_NAS-Bench-201 - CIFAR100.pkl"  # Replace <exp_name> with the actual experiment name
    # file_path = "./p201-0/ImageNet16-120/predictors/gcn/search/acc_0516_c10/20240703-112813/trajectories_merge_NAS-Bench-201 - ImageNet16-120.pkl"  # Replace <exp_name> with the actual experiment name
    # 301
    # search_space = 'nasbench301'
    # dataset = 'cifar10'  # cifar10 for NB101 and NB301, {cifar100, ImageNet16-120} for NB201
    # file_path = "./p301-0/cifar10/predictors/gcn/search/test_301/20240702-112511/trajectories_merge_NAS-Bench-301.pkl"  # Replace <exp_name> with the actual experiment name
    # nlp
    search_space = 'nlp'
    dataset = 'ptb'  # cifar10 for NB101 and NB301, {cifar100, ImageNet16-120} for NB201
    file_path = "./pnlp-0/ptb/predictors/gcn/search/test_nlp/20240702-153108/trajectories_merge_NAS-Bench-NLP.pkl"  # Replace <exp_name> with the actual experiment name

    with open(file_path, 'rb') as f:
        trajectories_merge = pickle.load(f)
    directory_path = os.path.dirname(file_path)
    config = utils.load_config(path=yaml_config[search_space][dataset]['file'])
    # config = get_config_from_args(config_file=yaml_config[search_space][dataset]['file'], config_type="st")

    plot_optimizers(search_space, dataset, trajectories_merge, yaml_config[search_space][dataset]['exp_name'], optimum_value=float(config.max_acc), x_runtime=False, save = directory_path)
