
# Setup

First, create a conda environment.

```bash
conda create -n efn python=3.8.18
conda activate efn
pip install -r requirements.txt
```

We have confirmed the feasibility of the environment setup in a new environment. If there are any environmental issues, you can refer to the installation dependencies of the NASLib library.

For `NAS-Bench-301` and `NAS-Bench-NLP`, additionally, you will have to install the NAS-Bench-301 API from [here](https://github.com/crwhite14/nasbench301).

## Benchmarks
Prepare the benchmark data files from the these URLs and place them in `naslib/data`.

| Benchmark     | Task                               | Datasets | Data URL                                                                                                                                                                                                                                                                                              | Required Files                                                                                                                              |
|---------------|------------------------------------|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
|NAS-Bench-101  | Image Classification   |                    CIFAR10                   | [cifar10](https://drive.google.com/file/d/1oORtEmzyfG1GcnPHh0ijCs0gCHKEThNx/view?usp=sharing)                                                                                                                                                                                                         | `naslib/data/nasbench_only108.pkl`                                                                                                          |
|NAS-Bench-201  | Image Classification   |  CIFAR10 <br> CIFAR100 <br>ImageNet16-120   | [V1.0](https://drive.google.com/file/d/1SKW0Cu0u8-gb18zDpaAGi0f74UdXeGKs/view) | For convenience, we have extracted a lightweight dataset file (only with final val and test accuracy) for the experiments, which will be submitted along with the code repository. <br />naslib/data/nb201_cifar10-valid_val_test_v1_0.pickle<br />naslib/data/nb201_cifar100_val_test_v1_0.pickle<br />naslib/data/nb201_ImageNet16-120_val_test_v1_0.pickle<br/>Our extraction code is naslib/data/convert_201.py, with the preparation of origial NB201 pickle file: [V1.0](https://drive.google.com/file/d/1SKW0Cu0u8-gb18zDpaAGi0f74UdXeGKs/view)<br /><br />Yet, you may interested in running with the original benchmark dataset provided in naslib:<br />[cifar10](https://drive.google.com/file/d/1sh8pEhdrgZ97-VFBVL94rI36gedExVgJ/view?usp=sharing) : naslib/data/nb201_cifar100_full_training.pickle<br />[cifar100](https://drive.google.com/file/d/1hV6-mCUKInIK1iqZ0jfBkcKaFmftlBtp/view?usp=sharing) : naslib/data/nb201_cifar10_full_training.pickle<br/> [imagenet](https://drive.google.com/file/d/1FVCn54aQwD6X6NazaIZ_yjhj47mOGdIH/view?usp=sharing): naslib/data/nb201_ImageNet16_full_training.pickle <br />However, it should be noted that these pickle files may not be extracted based on the mean value of multiple seeds and do not comply with our experimental setup. |
|NAS-Bench-301  | Image Classification   |                    CIFAR10                   | [cifar10](https://drive.google.com/file/d/1YJ80Twt9g8Gaf8mMgzK-f5hWaVFPlECF/view?usp=sharing)<br> [nb_models_v0.9](https://figshare.com/articles/software/nasbench301_models_v0_9_zip/12962432)       | `naslib/data/nb301_full_training.pickle` <br> `naslib/data/nb_model_0.9` <br>                                                                             |
|NAS-Bench-NLP  | Natural Language Processing   |           Penn Treebank               | [ptb](https://drive.google.com/file/d/1DtrmuDODeV2w5kGcmcHcGj5JXf2qWg01/view?usp=sharing), [models](https://drive.google.com/file/d/13Kbn9VWHuBdSN3lG4Mbyr2-VdrTsfLfd/view?usp=sharing)                                                                                                               | `naslib/data/nb_nlp.pickle` <br> `naslib/data/nbnlp_v01/...`                                                                                |
| HW-NAS-Bench | Image Classification with Edge/Embedded/NPU Device constraints |           ImageNet16-120             | [dataset](https://github.com/GATECH-EIC/HW-NAS-Bench/blob/main/HW-NAS-Bench-v1_0.pickle)                                                                                                                                                                                                              | `naslib/search_spaces/HW_NAS_Bench/HW-NAS-Bench-v1_0.pickle` <br><br />NAS-Bench-201:<br />naslib/data/nb201_cifar10-valid_val_test_v1_0.pickle<br />naslib/data/nb201_cifar100_val_test_v1_0.pickle<br />naslib/data/nb201_ImageNet16-120_val_test_v1_0.pickle<br/><br /> |

Add execute permission to the scripts:

```bash
chmod +x -R ./scripts/nas/
```

To run NAS-Bench-101

```bash
bash ./scripts/nas/run_101st.sh
```
To run NAS-Bench-201
```bash
bash ./scripts/nas/run_201st_c10.sh
bash ./scripts/nas/run_201st_c100.sh
bash ./scripts/nas/run_201st_in_100.sh
bash ./scripts/nas/run_201st_in_190.sh
bash ./scripts/nas/run_201st_in_280.sh
```
To run NAS-Bench-301
```bash
bash ./scripts/nas/run_301st_100.sh
bash ./scripts/nas/run_301st_50.sh
```

To run NAS-Bench-NLP

```bash
bash ./scripts/nas/run_nlpst_150.sh
bash ./scripts/nas/run_nlpst_75.sh
```



To run HW-NAS-Bench (NAS-Bench-201 search space), the accuracy values are from the 201 pickle file. For convenience and data load efficiency, we have prepared a script to extract a new lightweight pickle file (each of approximately 30 MB) from the original pickle file. Running this script will overwrite the files (approximately 2 MB) that we provided.

```bash
python ./naslib/data/convert_hw_low_fi.py
```

We run 12 tasks on HW-NAS-Bench:

```bash
bash ./scripts/nas/run_hwst_in.sh /path/to/the/yaml
```

Our yamls can be found in ./hw_nas_bench/yaml



To run low_fidelity experiments, for convenience and data load efficiency, run convert script to extract a new pickle file (each of approximately 30 MB) from the original pickle file. 

```bash
python ./naslib/data/convert_hw_low_fi.py
```
We run multiple low fidelity experiments on NAS-Bench-201:

```bash
bash ./scripts/nas/run_low_fi.sh /path/to/the/yaml
```

Our yamls can be found in ./low_fi_nas_bench/yamls, you need to manually process the results. 



To run query efficiency test experiments，and specify the search space and dataset:

```bash
export PYTHONPATH=./
python naslib/runners/predictors/run_nas_optimizer.py --search_space nasbench101 --dataset cifar10
```

and draw the figures by script (need to modify the search trajectories files): 

```
python plt_query_efficient.py
```



Our codebase is fundamentally based on the [NASLib](https://github.com/automl/NASLib/tree/Develop). Currently, to maintain compatibility and ensure fairness in search comparisons, we have adopted its original *arch* data structure. However, we have found that this impacts the running time of benchmark query based search tests, especially since timestamps reveal that each query requires an *arch.clone()* operation, which is time-consuming. We plan to decouple or resolve this issue when releasing the open-source code.
