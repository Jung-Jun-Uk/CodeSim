lang=python
# mkdir -p ./saved_models/$lang
python run.py \
    --project=work_dir \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --lang=$lang \
    --do_submit \
    --code_folder=code/ \
    --num_train_epochs 10 \
    --code_length 512 \
    --data_flow_length 128 \
    --nl_length 128 \
    --batch_size 512 \
    --learning_rate 2e-4 \
    --head arcface \
    --aux upg_ff_fw \
    --name batch256_epoch_20_s64_wisk1.0_sp_sn_m0.5_class2_sampler_re2_dropout0.1 \
    --reverse_tokens \
    --device 4,5,6,7 \
    --seed 123456