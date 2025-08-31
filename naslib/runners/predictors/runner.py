import logging

from naslib.defaults.predictor_evaluator import PredictorEvaluator
from naslib.utils.encodings import EncodingType

from naslib.predictors import (
    GCNPredictor,
)

from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces import (
    NasBench101SearchSpace,
    NasBench201SearchSpace,
    NasBench301SearchSpace,
    NasBenchNLPSearchSpace,
    TransBench101SearchSpaceMicro,
    TransBench101SearchSpaceMacro,
    NasBenchASRSearchSpace,
)
from naslib import utils
from naslib.utils import setup_logger, get_dataset_api

config, args = utils.get_config_from_args(config_type="predictor")
utils.set_seed(config.seed)
logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)
utils.log_args(config)

supported_predictors = {

    "gcn": GCNPredictor(encoding_type=EncodingType.GCN, hpo_wrapper=True, config=config),
}

supported_search_spaces = {
    "nasbench101": NasBench101SearchSpace(),
    "nasbench201": NasBench201SearchSpace(),
    "nasbench301": NasBench301SearchSpace(),
    "nlp": NasBenchNLPSearchSpace(),
    'transbench101_micro': TransBench101SearchSpaceMicro(config.dataset),
    'transbench101_macro': TransBench101SearchSpaceMacro(),
    "asr": NasBenchASRSearchSpace(),
}

"""
If the API did not evaluate *all* architectures in the search space, 
set load_labeled=True
"""
load_labeled = True if config.search_space in ["nasbench301", "nlp"] else False
dataset_api = get_dataset_api(config.search_space, config.dataset)

if config.search_space == "nasbench101":
    acc = 0.0
    max_str = ""
    for item, value in dataset_api['nb101_data'].computed_statistics.items():
        for dict_list in value.values():
            for l in dict_list:
                if float(l['final_validation_accuracy']) > acc:
                    acc = float(l['final_validation_accuracy'])
                    max_str = item
    print(f'acc = {acc}, max_str = {max_str}')

predictor = supported_predictors[config.predictor]
search_space = supported_search_spaces[config.search_space]


predictor_evaluator = PredictorEvaluator(predictor, config=config)
predictor_evaluator.adapt_search_space(
    search_space, load_labeled=load_labeled, dataset_api=dataset_api
)


predictor_evaluator.evaluate()
