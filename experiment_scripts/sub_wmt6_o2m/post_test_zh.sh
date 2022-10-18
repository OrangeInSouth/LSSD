# usage: bash post_test_zh.sh ${checkpoint_path}/test_en_zh.txt

result_file=$1

cat $result_file | grep -P "^H" | sort -V | cut -f 3-  > $result_file.hyp.detok
cat $result_file | grep -P "^T" | sort -V | cut -f 2-  > $result_file.ref.detok

cat $result_file.hyp.detok | sacrebleu -w 2 -l en-zh $result_file.ref.detok > $result_file.score
cat $result_file.score