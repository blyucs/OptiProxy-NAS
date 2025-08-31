export OMP_NUM_THREADS=1

cur_project_dir="$(pwd)"
echo "Current project path: $cur_project_dir"

cur_script_dir="$(dirname "$(realpath "$0")")"
echo "Script path: $cur_script_dir"

cd "$cur_cur_project_dir"
pwd





start_seed=$1
if [ -z "$start_seed" ]
then
    start_seed=0
fi

base_file=naslib
s3_folder=nas201
out_dir=$s3_folder\_$start_seed


search_space=nasbench201
dataset=ImageNet16-120


trials=10
test_id=0
current_time=$(date "+%Y%m%d_%H%M%S")
config_name="${current_time}_test_fix_bug_adj"
end_seed=$(($start_seed + $trials - 1))

cp -f $cur_project_dir/p201-0/$dataset/configs/predictors/config_gcn_search_train_190.yaml $cur_project_dir/p201-0/$dataset/configs/predictors/config_gcn_search_train.yaml


for i in $(seq $start_seed $end_seed)
do
  python $cur_script_dir/create_yaml.py $i $test_id $config_name $cur_project_dir/p201-0/$dataset
done

for t in $(seq $start_seed $end_seed)
do

  sleep 2
  config_file=$cur_project_dir/p201-0/$dataset/configs/predictors/yamls/config\_gcn\_search\_train\_$t.yaml

  python $cur_project_dir/naslib/runners/predictors/search_train.py --config-file $config_file &

done
wait
results_dir="$cur_project_dir/p201-0/$dataset/predictors/gcn/search/$config_name"
python $cur_script_dir/result_analysis.py $results_dir
exit