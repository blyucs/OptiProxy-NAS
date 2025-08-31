optimizer=npenas
#predictors=(omni_seminas bananas mlp lgb gcn bonas xgb ngb rf dngo \
#bohamiann bayes_lin_reg gp seminas sparse_gp var_sparse_gp nao)
predictors=(mlp gcn gp)
start_seed=$1
if [ -z "$start_seed" ]
then
    start_seed=0
fi

# folders:
base_file=naslib
s3_folder=np201
out_dir=$s3_folder\_$start_seed

# search space / data:
search_space=nasbench201
dataset=cifar10
search_epochs=100

# trials / seeds:
trials=10
end_seed=$(($start_seed + $trials - 1))
save_to_s3=false

# create config files
for i in $(seq 0 $((${#predictors[@]}-1)) )
do
    predictor=${predictors[$i]}
    python scripts/create_configs.py --predictor $predictor \
    --epochs $search_epochs --start_seed $start_seed --trials $trials \
    --out_dir $out_dir --dataset=$dataset --config_type nas_predictor \
    --search_space $search_space --optimizer $optimizer
done

# run experiments
for t in $(seq $start_seed $end_seed)
do
    for predictor in ${predictors[@]}
    do
        config_file=$out_dir/$dataset/configs/nas_predictors/config\_$optimizer\_$predictor\_$t.yaml
        echo ================running $predictor trial: $t =====================
        python $base_file/runners/nas_predictors/runner.py --config-file $config_file
    done
#    if [ "save_to_s3" ]
#    then
#        # zip and save to s3
#        echo zipping and saving to s3
#        zip -r $out_dir.zip $out_dir
#        python $base_file/benchmarks/upload_to_s3.py --out_dir $out_dir --s3_folder $s3_folder
#    fi
done
