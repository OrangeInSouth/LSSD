# Language-Specific Self-Distillation
Code for the paper [Unifying the Convergences in Multilingual Neural Machine Translation](https://arxiv.org/abs/2205.01620).

This paper propose a novel training strategy named LSSD (*Language-Specific Self-Distillation*) remedying the convergence inconsistency in multilingual neural machine translation.

## Install
### 1. Downloading this repository.
```git clone``` or downloading the zip directly.
### 2. Installing the dependency.
```pip install -r requirements.txt```

and

```pip install --editable ./ ```

### 3. Downloading the data file (data-bin)
Downloading the [data file](https://drive.google.com/drive/folders/1z396pP8ZfCeiJm-CMIu9EdQqFpfckAZP) and unzip these files into a user-specific data-bin path.
We conducted experiments on three datasets (TED-8-Diverse, TED-8-Related, and WMT-6).
The TED-8-Diverse and TED-8-Related are provided by [1]. And we added language-specific dicts additionally.

### 4. Training and Inference.
Scripts for training and inference are placed in the 'experiment_scripts'. 

Before running these scripts, **modifying the project_path, python_path, path_2_data** as:
* ```project_path```: refers to the current project.
* ```python_path```: refers to the path of python (you can obtain it by ```which python```)
* ```path_2_data```: refers to the directory of data-bin files.

## Citation

Please cite as:

```bibtex
@inproceedings{huang-etal-2022-unifying,
  title = {Unifying the Convergences in Multilingual Neural Machine Translation},
  author = {Yichong Huang, Xiaocheng Feng, Xinwei Geng, Bing Qin},
  booktitle = {EMNLP},
  year = {2022},
}
```

## References
[1] Xinyi Wang, Yulia Tsvetkov, and Graham Neubig. 2020. Balancing training for multilingual neural machine translation. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 8526–8537, Online. Association for Computational Linguistics.