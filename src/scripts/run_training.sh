cd /home/krzysztof/studia/magisterka/time-series-invariance/TS-TCC
conda activate TS-TCC

timestamp=$(date +%Y-%m-%d_%H-%M-%S)
run_id=timestamp

for dataset in ../data/datasets/*; do
    dataset_name=$(basename "$dataset")
    dataset_name="${dataset_name%.*}"
    echo "Training on dataset: $dataset_name"
    python main.py \
        --experiment_description "exp_$dataset_name" \
        --run_description "run_$run_id" \
        --seed 123 \
        --selected_dataset $dataset_name \
        --device cpu \
        --training_mode self_supervised || break
    python main.py \
        --experiment_description "exp_$dataset_name" \
        --run_description "run_$run_id" \
        --seed 123 \
        --selected_dataset $dataset_name \
        --device cpu \
        --training_mode fine_tune || break
done

cd /home/krzysztof/studia/magisterka/time-series-invariance