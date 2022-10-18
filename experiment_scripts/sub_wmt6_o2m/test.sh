
python_path=/userhome/anaconda3/envs/ychuang_online_distillation/bin/python
project_path="/userhome/ychuang/online_distillation_MNMT"
lang_pairs="en-fr,en-de,en-zh,en-et,en-ro,en-tr"
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

for src in en; do
#    for src in bos; do
    for tgt in fr de zh et ro tr; do
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

python ${project_path}/experiment_scripts/sub_wmt6_o2m/result_statistics.py $OUTPUT_DIR > ${OUTPUT_DIR}/result.txt 2>&1
bash ${project_path}/experiment_scripts/sub_wmt6_o2m/post_test_zh.sh ${OUTPUT_DIR}/test_en_zh.txt