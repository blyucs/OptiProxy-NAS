import yaml
import sys
import os



seed = sys.argv[1]
test_id = sys.argv[2]
config_name = sys.argv[3]
base_path = sys.argv[4]


input_file_path = base_path + "/configs/predictors/config_gcn_search_train.yaml"
output_file_path = base_path + "/configs/predictors/yamls/config_gcn_search_train_" + seed + ".yaml"
directory = base_path + "/configs/predictors/yamls/"

# Check if the directory exists
if not os.path.exists(directory):
    # If it doesn't exist, create it
    os.makedirs(directory)

with open(input_file_path) as stream:
    try:
        src = yaml.safe_load(stream)
        with open(output_file_path, 'w') as stream2:
            try:
                src['seed'] = int(seed)
                src['test_id'] = str(test_id)
                src['config_name'] = config_name
                yaml.safe_dump(src, stream2)
            except yaml.YAMLError as exc:
                print(exc)
    except yaml.YAMLError as exc:
        print(exc)
