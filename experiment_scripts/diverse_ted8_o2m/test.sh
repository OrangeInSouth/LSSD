
python_path="/userhome/anaconda3/envs/ychuang_online_distillation/bin/python"
project_path="/userhome/ychuang/online_distillation_MNMT"
lang_pairs="eng-bos,eng-mar,eng-hin,eng-mkd,eng-ell,eng-bul,eng-fra,eng-kor"
path_2_data=${project_path}/data-bin/ted_8_diverse
lang_list=${project_path}/lang_list_diverse.txt

checkpoint_path=$1
model=${checkpoint_path}/checkpoint_best.pt

OUTPUT_DIR=$checkpoint_path
#TEST_DIR=/users6/chye/working/corpus/iwslt17/data

mkdir -p $OUTPUT_DIR

# echo "results" > $OUTPUT_DIR/results.txt

# CUDA_VISIBLE_DEVICES=0
for src in eng; do
#    for src in bos; do
    for tgt in bos mar hin mkd ell bul fra kor; do
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

#
python ${project_path}/experiment_scripts/diverse_ted8_o2m/result_statistics.py $OUTPUT_DIR > ${OUTPUT_DIR}/result.txt 2>&1