
# import mlperf phase1 checkpoint into nvidia-bert+ort

TF_CKP=../mlperf-bert/bs64k_32k_ckpt
BERT_PREP_WORKING_DIR=../mlperf-bert/data
bs=256

python run_pretraining_ort.py \
         --config_file $TF_CKP/bs64k_32k_ckpt_bert_config.json \
         --input_dir $BERT_PREP_WORKING_DIR/hdf5_lower_case_1_seq_len_512_max_pred_76_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en \
         --output_dir ../mlperf-bert/output/test/0/ \
         --bert_model   bert-large-uncased \
         --train_batch_size $bs \
         --max_seq_length 512 \
         --max_predictions_per_seq 76  \
         --max_steps 1500  \
         --warmup_proportion  0.128 \
         --num_steps_per_checkpoint 500 \
         --learning_rate 1e-3  \
         --seed 42 \
         --gradient_accumulation_steps 16 \
         --allreduce_post_accumulation \
         --allreduce_post_accumulation_fp16  \
         --do_train \
         --phase2 \
         --fp16 \
         --phase1_end_step 7038 \
         --resume_from_checkpoint \
         --disable_progress_bar \
         --init_checkpoint_tf $TF_CKP/bs64k_32k_ckpt_model.ckpt-28252 \
         --use_ib \
         --gpu_memory_limit_gb 32 \
         $@
