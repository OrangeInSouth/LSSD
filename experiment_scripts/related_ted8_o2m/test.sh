#!/usr/bin/env bash

python_path=/userhome/anaconda3/envs/ychuang_online_distillation/bin/python
project_path="/userhome/ychuang/online_distillation_MNMT"
lang_pairs="eng-aze,eng-bel,eng-glg,eng-slk,eng-tur,eng-rus,eng-por,eng-ces"
path_2_data=${project_path}/data-bin/ted_8_related
lang_list=${project_path}/lang_list_related.txt

checkpoint_path=$1
model=${checkpoint_path}/checkpoint_best.pt

OUTPUT_DIR=$checkpoint_path

mkdir -p $OUTPUT_DIR

for src in eng; do
    for tgt in aze bel glg slk tur rus por ces; do
        ${python_path} ${project_path}/fairseq_cli/generate.py $path_2_data \
            --path $model \
            --task translation_multi_simple_epoch \
            --lang-dict "$lang_list" \
            --lang-pairs "$lang_pairs" \
            --gen-subset test \
            --source-lang $src \
            --target-lang $tgt \
            --encoder-langtok "tgt" \
            --scoring sacrebleu \
            --remove-bpe 'sentencepiece'\
            --batch-size 96 \
            --decoder-langtok > $OUTPUT_DIR/test_${src}_${tgt}.txt 2>&1

    done
done

python ${project_path}/experiment_scripts/related_ted8_o2m/result_statistics.py $OUTPUT_DIR > ${OUTPUT_DIR}/result.txt 2>&1