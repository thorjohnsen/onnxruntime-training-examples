
# run ort training

TF_CKP=/workspace/phase1
BERT_PREP_WORKING_DIR=/data/512
bs=4096

python -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 run_pretraining_ort.py \
         --config_file $TF_CKP/bert_config.json \
         --input_dir $BERT_PREP_WORKING_DIR \
         --output_dir /results \
         --bert_model   bert-large-uncased \
         --train_batch_size $bs \
         --max_seq_length 512 \
         --max_predictions_per_seq 76  \
         --max_steps 1500  \
         --warmup_proportion  0.128 \
         --num_steps_per_checkpoint 500 \
         --learning_rate 1e-3  \
         --seed 42 \
         --gradient_accumulation_steps 256 \
         --allreduce_post_accumulation \
         --allreduce_post_accumulation_fp16  \
         --do_train \
         --phase2 \
         --fp16 \
         --phase1_end_step 7038 \
         --resume_from_checkpoint \
         --disable_progress_bar \
         --init_checkpoint $TF_CKP/model.ckpt-28252.pt \
         --use_ib \
         --gpu_memory_limit_gb 32 \
         $@
