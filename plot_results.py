# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
from pathlib import Path
import json
import os

plt.rcParams.update({'text.color': "black",
                     'axes.labelcolor': "black"})

# %%
'''Config'''
experiments_sequence = 2
if experiments_sequence == 1:
    '''Draw an exp_gamma plot'''
    directory_start = Path("/home/uberariy/dte-stand/data_examples/rhombus/")
    experiments = {
        "exp_gamma = 0.00": "TWOL_rhombus_10M20M_memmory_size256_threshold0.01_storelinkstates_expgamma0.00",
        "exp_gamma = 0.20": "TWOL_rhombus_10M20M_memmory_size256_threshold0.01_storelinkstates_expgamma0.20",
        "exp_gamma = 0.40": "TWOL_rhombus_10M20M_memmory_size256_threshold0.01_storelinkstates_expgamma0.40",
        "exp_gamma = 0.60": "TWOL_rhombus_10M20M_memmory_size256_threshold0.01_storelinkstates_expgamma0.60",
        "exp_gamma = 0.80": "TWOL_rhombus_10M20M_memmory_size256_threshold0.01_storelinkstates_expgamma0.80",
        "exp_gamma = 1.00": "TWOL_rhombus_10M20M_memmory_size256_threshold0.01_storelinkstates_expgamma1.00"
    }
elif experiments_sequence == 2:
    '''Draw a memory-size plot'''
    directory_start = Path("/home/uberariy/maroh/exp_sequence_2")
    names = [
        "TWOL_rhombus_10M20M_memmory_size64_threshold0.01_storelinkstates_expgamma1.00",
        "TWOL_rhombus_10M20M_memmory_size128_threshold0.01_storelinkstates_expgamma1.00",
        "TWOL_rhombus_10M20M_memmory_size256_threshold0.01_storelinkstates_expgamma1.00",
        "TWOL_rhombus_10M20M_memmory_size512_threshold0.01_storelinkstates_expgamma1.00",
        "TWOL_rhombus_10M20M_memmory_size1024_threshold0.01_storelinkstates_expgamma1.00",
        "TWOL_rhombus_10M20M_memmory_size0_threshold0.01_storelinkstates_expgamma1.00",
        # "TWOL_rhombus_10M20M_memmory_size-1_threshold0.01_storelinkstates_expgamma1.00", # Equal to size0, but have better dropout
    ]
    # names = [
    #     "TWOL_rhombus_BASIC_memmory_size0_threshold0.01_storelinkstates_expgamma1.00_removedropout"
    # ]
    experiment_labels = dict()
    for name in names:
        experiment_labels[name] = dict()
        features = ["size", "threshold", "expgamma"]
        for n in name.split("_"):
            for f in features:
                if n.startswith(f):
                    experiment_labels[name][f] = float(n[len(f):])
    hue_feature = "size"
elif experiments_sequence == 3:
    '''Draw a memory-size plot'''
    directory_start = Path("/home/uberariy/maroh/exp_sequence_3_check_multiple_actions")
    names = [
        "etalon_one_action",
        "run_with_weights_output_actorloss100",
        "run_with_weights_output_loss50",
        "run_with_weights_output_multiple_actions_2",
        "run_with_weights_output_entropyloss0.01_criticloss0.01_greedyeps0.5",
        "run_with_weights_output_entropyloss0.01_criticloss0.01_greedyeps0.9",
        "run_with_weights_output_entropyloss0.01_criticloss0.01_greedyepsuniform0.5"
    ]
    experiment_labels = dict()
    for name in names:
        experiment_labels[name] = dict()
        experiment_labels[name]["name"] = name
    hue_feature = "name"

# %%
def parse_experiment_data(directory_start, experiment_labels):

    phis = dict()
    message_ratios = dict()
    exp_properties = dict()

    for exp_dir in os.listdir(directory_start):
        # filename = os.fsdecode(exp_dir)
        # print(exp_dir[20:])
        if exp_dir[20:] not in experiment_labels:
            continue
        exp_properties[exp_dir] = experiment_labels[exp_dir[20:]]
        phis[exp_dir] = defaultdict(list)
        message_ratios[exp_dir] = defaultdict(list)
        for filename in os.listdir(directory_start / exp_dir / "iteration0"):
            if not filename.endswith(".json"): 
                continue
            tmp = filename[:-5].split('_')[-1]
            with open(directory_start / exp_dir / "iteration0" / filename) as f:
                data = json.load(f)
            if filename.startswith("phi_values"):
                mea = [sum(d) / len(d) for d in data]
                mea2 = sum(mea) / len(mea)
                phis[exp_dir]["y"].append(mea2)
                phis[exp_dir]["x"].append(int(tmp.split('-')[0]))
            if filename.startswith("messages"):
                if exp_dir[20:] == "TWOL_rhombus_10M20M_memmory_size0_threshold0.01_storelinkstates_expgamma1.00":
                    message_ratios[exp_dir]["y"].append((int(tmp.split('-')[0]) + 200) * 800)
                    message_ratios[exp_dir]["x"].append(int(tmp.split('-')[0]))
                    continue
                message_ratios[exp_dir]["y"].append(data["message_iterations_done"][-1])
                message_ratios[exp_dir]["x"].append(int(tmp.split('-')[0]))
            # print(filename[:-5].split('_')[-1], )
        sort = np.argsort(phis[exp_dir]["x"])
        phis[exp_dir]["y"] = np.array(phis[exp_dir]["y"])[sort]
        phis[exp_dir]["x"] = np.sort(np.array(phis[exp_dir]["x"]))
        sort = np.argsort(message_ratios[exp_dir]["x"])
        message_ratios[exp_dir]["y"] = np.array(message_ratios[exp_dir]["y"])[sort]
        message_ratios[exp_dir]["x"] = np.sort(np.array(message_ratios[exp_dir]["x"]))
    return phis, message_ratios, exp_properties

def calculate_means_in_lists(d):
    d2 = dict()
    for a, b in d.items():
        if len(b) == 1:
            d2[a] = b[0]
        else:
            lens = [len(m) for m in b]
            d2[a] = []
            for i in range(max(lens)):
                tmp = [m[i] for j, m in enumerate(b) if lens[j] >= i + 1]
                d2[a].append(sum(tmp) / len(tmp))
    return d2


# %%
phis, message_ratios, exp_properties = parse_experiment_data(directory_start=directory_start, experiment_labels=experiment_labels)

# %%
if experiments_sequence == 1:
    pass
elif experiments_sequence >= 2:
    ogranichenie = 30
    phis_m = dict()
    message_ratios_m = dict()
    for exp_name, properties in exp_properties.items():
        cur_label = hue_feature + ": " + str(properties[hue_feature])
        if cur_label in phis_m.keys():
            phis_m[cur_label].append(phis[exp_name]["y"])
            message_ratios_m[cur_label].append(message_ratios[exp_name]["y"])
        else:
            phis_m[cur_label] = [phis[exp_name]["y"]]
            message_ratios_m[cur_label] = [message_ratios[exp_name]["y"]]
        ygrecs = np.linspace(phis[exp_name]["x"][1], ogranichenie * phis[exp_name]["x"][1], ogranichenie)
    phis_m = calculate_means_in_lists(phis_m)
    message_ratios_m = calculate_means_in_lists(message_ratios_m)
    # print(phis_m)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(22, 20))
    fig.suptitle("Граф целевой функции & доли итераций обмена сообщений")
    # Histogram Plot of Observed Data
    for hue_feature_v, exp_path in phis_m.items():
        # print(phis[exp_name]["x"], phis[exp_name]["y"])
        tmp = phis_m[hue_feature_v][:ogranichenie]
        axes[0].plot(ygrecs[:len(tmp)], tmp, label=hue_feature_v)
    axes[0].legend()
    axes[0].set_title("Зависимость значения целевой функции от эпизода")
    axes[0].set_xlabel('Эпизод')
    axes[0].set_ylabel('Значение целевой функции')

    for hue_feature_v, exp_path in message_ratios_m.items():
        # print(message_ratios[exp_name]["x"], message_ratios[exp_name]["y"])
        tmp = message_ratios_m[hue_feature_v][:ogranichenie]
        axes[1].plot(ygrecs[:len(tmp)], tmp, label=hue_feature_v)
    axes[1].set_title("Зависимость количества коммуникаций от эпизода")
    axes[1].set_xlabel('Эпизод')
    axes[1].set_ylabel('Количество действий коммуникации')

# %%
if experiments_sequence == 2:
    # Mean phis:
    print([(a, sum(b) / len(b)) for a, b in phis_m.items()])
    # Message ratios:
    print([(a, b[-1] / 4000000) for a, b in message_ratios_m.items()])

# %%
