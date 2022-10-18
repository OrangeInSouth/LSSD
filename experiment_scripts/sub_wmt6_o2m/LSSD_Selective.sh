
python_path=/userhome/anaconda3/envs/ychuang_online_distillation/bin/python
project_path="/userhome/ychuang/online_distillation_MNMT"
lang_pairs="en-fr,en-de,en-zh,en-et,en-ro,en-tr"
path_2_data=/userhome/ychuang/datasets/data-bin/WMT6-small
lang_list=${path_2_data}/lang_list.txt

SAVE_DIR=${project_path}/checkpoints/sub_wmt6_o2m/LSSD_Selective
mkdir -vp $SAVE_DIR

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
${python_path} ${project_path}/fairseq_cli/train.py $path_2_data \
  --save-dir $SAVE_DIR \
  --encoder-normalize-before --decoder-normalize-before \
  --arch transformer_wmt_en_de --layernorm-embedding \
  --task translation_multi_simple_epoch \
  --sampling-method "temperature" \
  --sampling-temperature 5 \
  --decoder-langtok \
  --encoder-langtok "tgt" \
  --lang-dict "$lang_list" \
  --lang-pairs "$lang_pairs" \
  --label-smoothing 0.1 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 4000 --lr 0.0015 \
  --share-decoder-input-output-embed \
  --max-epoch 157 \
  --dropout 0.3 --weight-decay 0.0 \
  --max-tokens 8192 --update-freq 2 \
  --save-interval 1 \
  --seed 222 --log-format simple --log-interval 10 \
  --bpe sentencepiece \
  --fp16 \
  --pure-batch \
  --LS-epoch \
  --online-distillation-MNMT \
  --language-aware-online-distillation \
  --criterion label_smoothed_cross_entropy_with_online_distillation \
  --online-distillation-weight 0.8 \
  --selective-online-distillation "hard" \
  --selective-online-distillation-level sentence > ${SAVE_DIR}/train.log 2>&1

bash ${project_path}/experiment_scripts/sub_wmt6_o2m/test.sh $SAVE_DIR
