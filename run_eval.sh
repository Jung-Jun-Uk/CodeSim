lang=python
# mkdir -p ./saved_models/$lang
python run.py \
    --project=work_dir \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --lang=$lang \
    --do_eval \
    --code_folder=code/ \
    --num_train_epochs 20 \
    --code_length 512 \
    --data_flow_length 128 \
    --nl_length 128 \
    --batch_size 256 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --head arcface \
    --aux upg_ff_fw \
    --name batch256_epoch_20_s64_wisk1.0_sp_sn_m0.5_class2_sampler_re2_dropout0.1 \
    --seed 123456