#!/bin/sh
if [ "$1" = "diverse" ]; then
  langs=(bos mar hin mkd ell bul fra kor)
elif [ "$1" = "related" ]; then
  langs=(aze bel glg slk tur rus por ces)
elif [ "$1" = "wmt6" ]; then
  langs=(tr ro et zh de fr)
fi

res_path=$3
old_path=$(pwd)
cd ${res_path}

rm gen.txt
rm ref.txt
tmp="eng"
for lang in ${langs[@]}; do
  if [ "$2" = "M2O" ]; then
    file=test_${lang}_${tmp: 0 :${#lang}}.txt
  elif [ "$2" = "O2M" ]; then
    file=test_${tmp: 0 :${#lang}}_${lang}.txt
  fi

  echo $file
  cat $file | grep -P "^H" | sort -V | cut -f 3- >> gen.txt
  cat $file | grep -P "^T" | sort -V | cut -f 2- >> ref.txt
done

wc -l gen.txt
wc -l ref.txt

cd ${old_path}