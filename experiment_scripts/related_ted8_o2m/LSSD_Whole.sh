#!/usr/bin/env bash

python_path=/userhome/anaconda3/envs/ychuang_online_distillation/bin/python
project_path="/userhome/ychuang/online_distillation_MNMT"
lang_pairs="eng-aze,eng-bel,eng-glg,eng-slk,eng-tur,eng-rus,eng-por,eng-ces"
path_2_data=${project_path}/data-bin/ted_8_related
lang_list=${project_path}/lang_list_related.txt

SAVE_DIR=${project_path}/checkpoints/related_ted8_o2m/LSSD_Whole
mkdir -vp $SAVE_DIR

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
${python_path} ${project_path}/fairseq_cli/train.py $path_2_data \
  --save-dir $SAVE_DIR \
  --encoder-normalize-before --decoder-normalize-before \
  --arch transformer_iwslt_de_en --layernorm-embedding \
  --task translation_multi_simple_epoch \
  --sampling-method "temperature" \
  --sampling-temperature 1 \
  --decoder-langtok \
  --encoder-langtok "tgt" \
  --lang-dict "$lang_list" \
  --lang-pairs "$lang_pairs" \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 4000 --lr 0.002 \
  --share-decoder-input-output-embed \
  --max-epoch 600 --max-update 200000 \
  --dropout 0.3 --attention-dropout 0.3 --weight-decay 0.0 \
  --max-tokens 8192 --update-freq 1  \
  --save-interval 1 --save-interval-updates 7000  \
  --seed 222 --log-format simple --log-interval 1 \
  --bpe sentencepiece \
  --pure-batch \
  --LS-epoch \
  --online-distillation-MNMT \
  --criterion label_smoothed_cross_entropy_with_online_distillation --label-smoothing 0.1 \
  --online-distillation-weight 0.5 \
  --fp16 > ${SAVE_DIR}/train.log 2>&1

bash ${project_path}/experiment_scripts/related_ted8_m2o/test.sh $SAVE_DIR