
"""
This script aims to statistics some information about language-specific loss curves.
run example:
    python analyze_LS_loss_curve.py LS_valid_loss_history.json
"""

import sys
import math
import json


def get_convergence_epochs(loss_history):
    return loss_history.index(min(loss_history)) + 1


def get_best_loss(loss_history):
    return round(min(loss_history), 4)


def calculate_distance_to_upper_bound(history, epoch):
    best_loss = min(history)
    return round(history[epoch - 1] - best_loss, 4)


def count_distillation_num(loss_history):
    """
    This function aims at counting how many times online-distillation works.
    :param loss_history:
    :return:
    """
    best_loss = 10000
    best_epoch = -1
    distillation_nums = 0
    for i, v in enumerate(loss_history):
        epoch = i + 1
        if epoch > 1 and epoch - best_epoch > 1:
            distillation_nums += 1
        if v < best_loss:
            best_loss = v
            best_epoch = epoch
    return distillation_nums


def calculate_oscillation(loss_history):
    """
    This function aims to calculate the oscillation degree of loss curve.
    :param loss_history:
    :return:
    """
    assert len(loss_history) > 2
    first_derivate = []
    for i in range(1, len(loss_history)):
        first_derivate.append(loss_history[i] - loss_history[i - 1])

    second_derivate = []
    for i in range(1, len(first_derivate)):
        second_derivate.append(first_derivate[i] - first_derivate[i - 1])

    result = sum([math.fabs(i) for i in second_derivate])
    return round(result, 3)


if __name__ == "__main__":
    file_path = sys.argv[1]
    f = open(file_path)
    data = json.load(f)
    f.close()

    lang_pairs = list(data.keys())
    lang_pairs.remove("all")
    lang_pairs.append("all")
    lang_pair_num = len(lang_pairs)

    char_line = 106
    char_interval = (char_line - (lang_pair_num + 1)) // lang_pair_num

    for value, func in [("Convergence Epoch", get_convergence_epochs),
                        ("Best Loss", get_best_loss),
                        ("Distance to Upper-Bound", calculate_distance_to_upper_bound),
                        ("distillation num", count_distillation_num),
                        ("oscillation", calculate_oscillation)]:
        print('-'*char_line)
        print(f"|{value:^{char_line-2}}|")
        print('-' * char_line)
        print('|' + '|'.join([f"{lang_pair:^{char_interval}}" for lang_pair in lang_pairs]) + '|')
        print('-' * char_line)

        if value == "Distance to Upper-Bound":
            results = [func(data[lang_pair], get_convergence_epochs(data['all'])) for lang_pair in lang_pairs]
            results[-1] = round(sum(results[:-1]), 3)
            print('|' + '|'.join([f"{i:^{char_interval}}" for i in results]) + '|')
        else:
            print('|' + '|'.join([f"{func(data[lang_pair]):^{char_interval}}" for lang_pair in lang_pairs]) + '|')

        print('-' * char_line)
        print()
