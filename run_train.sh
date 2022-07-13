lang=python
# mkdir -p ./saved_models/$lang
python run.py \
    --project=work_dir \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --lang=$lang \
    --do_train \
    --do_submit \
    --code_folder=code/ \
    --num_train_epochs 100 \
    --code_length 512 \
    --data_flow_length 128 \
    --nl_length 128 \
    --batch_size 256 \
    --learning_rate 2e-4 \
    --max_grad_norm 1.0 \
    --head arcface \
    --aux upg_ff_fw \
    --reverse_tokens \
    --name batch256_epoch_100_s64_wisk_2.0_2.0_sp_sn_m0.5_class2_sampler_dropout_0.3_2e-4_reverse_tokens \
    --seed 123456
    