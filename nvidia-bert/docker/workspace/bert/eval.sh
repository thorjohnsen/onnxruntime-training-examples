
# eval mlm_acc from ort training

TF_CKP=../mlperf-bert/bs64k_32k_ckpt
BERT_PREP_WORKING_DIR=../mlperf-bert/data
DATA=$BERT_PREP_WORKING_DIR/test_dataset/hdf5_lower_case_1_seq_len_512_max_pred_76_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en

if [ $# -lt 1 ]; then
    echo "$0 path"
    exit 1
fi

for i in $*; do
    echo "====== $i ======"
    python eval.py \
            --config_file $TF_CKP/bs64k_32k_ckpt_bert_config.json  \
            --input_dir $DATA \
            --bert_model bert-large-uncased \
            --max_seq_length 512  \
            --max_predictions_per_seq 76 \
            --max_steps 156 \
            --eval_batch_size 64 \
            --seed 42 \
            --eval \
            --ckpt_path $i
done 
