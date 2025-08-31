import os
import pickle
import json
from pathlib import Path

from .asr import from_folder

"""
This file loads any dataset files or api's needed by the Trainer or PredictorEvaluator object.
They must be loaded outside of the search space object, because search spaces are copied many times
throughout the discrete NAS algos, which would lead to memory errors.
"""

def get_project_root() -> Path:
    """
    Returns the root path of the project.
    """
    return Path(__file__).parent.parent

def get_zc_benchmark_api(search_space, dataset):

    datafile_path = os.path.join(
        get_project_root(), "data", f"zc_{search_space}.json")
    with open(datafile_path) as f:
        data = json.load(f)

    return data[dataset]

def load_sampled_architectures(search_space, postfix=''):
    datafile_path = os.path.join(get_project_root(), "data", "archs", f"archs_{search_space}{postfix}.json")
    with open(datafile_path) as f:
        data = json.load(f)

    return data

def get_transbench101_api(dataset=None):
    datafile_path = os.path.join(
        get_project_root(), "data", "transnas-bench_v10141024.pth")
    assert os.path.exists(datafile_path), f"Could not fine {datafile_path}. Please download transnas-bench_v10141024.pth\
 from https://www.noahlab.com.hk/opensource/vega/page/doc.html?path=datasets/transnasbench101"

    from naslib.search_spaces import TransNASBenchAPI
    api = TransNASBenchAPI(datafile_path)
    return {'api': api, 'task': dataset}


def get_nasbench101_api(dataset=None):
    # load nasbench101
    import naslib.utils.nb101_api as api

    nb101_datapath = os.path.join(
        get_project_root(), "data", "nasbench_only108.pkl")
    assert os.path.exists(nb101_datapath), f"Could not find {nb101_datapath}. Please download nasbench_only108.pk \
from https://drive.google.com/drive/folders/1rwmkqyij3I24zn5GSO6fGv2mzdEfPIEa"

    nb101_data = api.NASBench(nb101_datapath)
    return {"api": api, "nb101_data": nb101_data}


def get_nasbench201_api(dataset=None, dataset_ver = 'v1_0'):
    """
    Load the NAS-Bench-201 data
    """
    # datafiles = {
    #     'cifar10': 'nb201_cifar10_full_training.pickle',
    #     'cifar100': 'nb201_cifar100_full_training.pickle',
    #     'ImageNet16-120': 'nb201_ImageNet16_full_training.pickle',
    #     'ninapro': 'nb201_ninapro_full_training.pickle'
    # }

    datafiles = {
        'cifar10': f'nb201_cifar10-valid_val_test_{dataset_ver}.pickle',
        'cifar100': f'nb201_cifar100_val_test_{dataset_ver}.pickle',
        'ImageNet16-120': f'nb201_ImageNet16-120_val_test_{dataset_ver}.pickle',
    }

    datafile_path = os.path.join(
        get_project_root(), 'data', datafiles[dataset])
    print(f"[[[[[[{datafile_path}")
    assert os.path.exists(datafile_path), f'Could not find {datafile_path}. Please download {datafiles[dataset]} from \
https://drive.google.com/drive/folders/1rwmkqyij3I24zn5GSO6fGv2mzdEfPIEa'

    with open(datafile_path, 'rb') as f:
        data = pickle.load(f)

    return {"nb201_data": data}


def get_nasbench301_api(dataset, dataset_ver = "v1.0"):
    if dataset != 'cifar10':
        return None
    # Load the nb301 performance and runtime models
    try:
        import nasbench301
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError('No module named \'nasbench301\'. \
            Please install nasbench301 from https://github.com/automl/nasbench301@no_gin using `pip install git+https://github.com/automl/nasbench301@no_gin`')

    # Paths to v1.0 model files and data file.
    download_path = os.path.join(get_project_root(), "data")
    if dataset_ver == 'v1.0':
        nb_models_path = os.path.join(download_path, "nb_models_1.0")
    elif dataset_ver == 'v0.9':
        nb_models_path = os.path.join(download_path, "nb_models_0.9")
    else:
        nb_models_path = os.path.join(download_path, "nb_models_0.9")
    os.makedirs(download_path, exist_ok=True)

    if dataset_ver == 'v1.0':
        nb301_model_path = os.path.join(nb_models_path, "xgb_v1.0")
    elif dataset_ver == 'v0.9':
        nb301_model_path = os.path.join(nb_models_path, "xgb_v0.9")
    else:
        nb301_model_path = os.path.join(nb_models_path, "xgb_v0.9")

    if dataset_ver == 'v1.0':
        nb301_runtime_path = os.path.join(nb_models_path, "lgb_runtime_v1.0")
    elif dataset_ver == 'v0.9':
        nb301_runtime_path = os.path.join(nb_models_path, "lgb_runtime_v0.9")
    else:
        nb301_runtime_path = os.path.join(nb_models_path, "lgb_runtime_v0.9")

    data_path = os.path.join(download_path, "nb301_full_training.pickle")

    if not all(os.path.exists(model) for model in [nb301_model_path,
                                                   nb301_runtime_path]):
        # nasbench301.download_models(version='1.0', delete_zip=True,
        #                             download_dir=download_path)
        print('please download nasbench301_models_{v0.9/v1.0}}.zip from and extract it to naslib/data/nb_model_{'
              '0.9/1.0}. Download nb301_full_training.pickle and put it in naslib/data/. The download link can be '
              'found in the README. \n So that, the structure looks like:\nnaslib\n-data\n--nb_models_{'
              '0.9/1.0}\n---gnn_gin_{v0.9/v1.0}\n---lgb_runtime_{v0.9/v1.0}\n---xgb_{'
              'v0.9/v1.0}\n--nb301_full_training.pickle\n\nExiting...')
        exit()

    models_not_found_msg = "Please download v1.0 models from \
https://figshare.com/articles/software/nasbench301_models_v1_0_zip/13061510"

    # Verify the model and data files exist
    assert os.path.exists(
        nb_models_path), f"Could not find {nb_models_path}. {models_not_found_msg}"
    assert os.path.exists(
        nb301_model_path), f"Could not find {nb301_model_path}. {models_not_found_msg}"
    assert os.path.exists(
        nb301_runtime_path), f"Could not find {nb301_runtime_path}. {models_not_found_msg}"

    performance_model = nasbench301.load_ensemble(nb301_model_path)
    runtime_model = nasbench301.load_ensemble(nb301_runtime_path)

    nb301_model = [performance_model, runtime_model]
    with open(data_path, "rb") as f:
        nb301_data = pickle.load(f)
        nb301_arches = list(nb301_data.keys())

    return {
        "nb301_data": nb301_data,
        "nb301_arches": nb301_arches,
        "nb301_model": nb301_model,
    }


def get_nlp_api(dataset=None):
    nb_model_path = os.path.join(get_project_root(), "data", "nbnlp_v01")
    nb_nlp_data_path = os.path.join(
        get_project_root(), "data", "nb_nlp.pickle")

    data_not_found_msg = "Please download the files from https://drive.google.com/drive/folders/1rwmkqyij3I24zn5GSO6fGv2mzdEfPIEa"

    assert os.path.exists(
        nb_model_path), f"Could not find {nb_model_path}. {data_not_found_msg}"
    assert os.path.exists(
        nb_nlp_data_path), f"Could not find {nb_nlp_data_path}. {data_not_found_msg}"

    # Load the NAS-Bench-NLP data
    with open(nb_nlp_data_path, "rb") as f:
        nlp_data = pickle.load(f)
    nlp_arches = list(nlp_data.keys())

    # Load the NAS-Bench-NLP11 performance model
    # try:
    #     import nasbench301
    # except ModuleNotFoundError as e:
    #     raise ModuleNotFoundError('No module named \'nasbench301\'. \
    #         Please install nasbench301 from https://github.com/crwhite14/nasbench301')
    from naslib.search_spaces.nasbenchx11.nas_bench_x11.api import load_ensemble
    # performance_model = nasbench301.load_ensemble(nb_model_path)
    performance_model = load_ensemble(nb_model_path)

    return {
        "nlp_data": nlp_data,
        "nlp_arches": nlp_arches,
        "nlp_model": performance_model,
    }


def get_asr_api(dataset=None):
    # Load the NAS-Bench-ASR data
    d = from_folder(os.path.join(get_project_root(), 'data'),
                    include_static_info=True)

    return {
        'asr_data': d,
    }


def get_natsbenchsize_api(dataset=None):
    try:
        from nats_bench import create
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError('No module named \'nats_bench\'. \
            Please install nats_bench from https://github.com/D-X-Y/NATS-Bench')

    # Create the API for size search space
    api = create(None, 'sss', fast_mode=True, verbose=True)
    return api


def get_dataset_api(search_space=None, dataset=None, dataset_ver=None):

    if search_space == "nasbench101":
        return get_nasbench101_api(dataset=dataset)

    elif search_space == "nasbench201":
        return get_nasbench201_api(dataset=dataset, dataset_ver = dataset_ver)

    elif search_space == "nasbench301":
        return get_nasbench301_api(dataset=dataset, dataset_ver = dataset_ver)

    elif search_space == "nlp":
        return get_nlp_api(dataset=dataset)

    elif search_space in ['transbench101', 'transbench101_micro', 'transbench101_macro']:
        return get_transbench101_api(dataset=dataset)

    elif search_space == "asr":
        return get_asr_api(dataset=dataset)

    elif search_space == 'natsbenchsize':
        return get_natsbenchsize_api(dataset=dataset)

    elif search_space == "test":
        return None

    else:
        raise NotImplementedError()
