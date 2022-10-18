#!/usr/bin/env bash

python_path=/userhome/anaconda3/envs/ychuang_online_distillation/bin/python
project_path="/userhome/ychuang/online_distillation_MNMT"
lang_pairs="fr-en,de-en,zh-en,et-en,ro-en,tr-en"
path_2_data=/userhome/ychuang/datasets/data-bin/WMT6-small
lang_list=${path_2_data}/lang_list.txt

checkpoint_path=$1
checkpoint_name=checkpoint_best.pt
if [ -n "$2" ]; then
  checkpoint_name=$2
fi
model=${checkpoint_path}/${checkpoint_name}
echo "model: ${model}"
OUTPUT_DIR=$checkpoint_path

mkdir -p $OUTPUT_DIR


for tgt in en; do
    for src in fr de zh et ro tr; do
        ${python_path} ${project_path}/fairseq_cli/generate.py $path_2_data \
            --path $model \
            --task translation_multi_simple_epoch \
            --lang-dict "$lang_list" \
            --lang-pairs "$lang_pairs" \
            --gen-subset test \
            --source-lang $src \
            --target-lang $tgt \
            --encoder-langtok "src" \
            --scoring sacrebleu \
            --remove-bpe 'sentencepiece'\
            --batch-size 96 \
            --decoder-langtok > $OUTPUT_DIR/test_${src}_${tgt}.txt 2>&1

    done
done

#
python ${project_path}/experiment_scripts/sub_wmt6_m2o/result_statistics.py $OUTPUT_DIR > ${OUTPUT_DIR}/result.txt 2>&1