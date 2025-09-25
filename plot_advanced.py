import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scienceplots
import json
from collections import defaultdict
import os
import re
import pandas as pd
from pathlib import Path
from datetime import datetime
import pandas as pd
import textwrap
from math import ceil

def parse_args():
    parser = argparse.ArgumentParser(description='Advanced plotting')
    parser.add_argument('experiments_list', type=Path,
                       help='path to .json file containing experiment directories and their labels, plot title, and optionally genetic algotithm results path')
    parser.add_argument('--ymin', type=float, default=None,
                       help='min value to display on y axis')
    parser.add_argument('--ymax', type=float, default=None,
                       help='max value to display on y axis')
    parser.add_argument('-n', '--episodes', type=str, default='max',
                       help="integer (number of episodes to display), or 'max' (display all episodes), or 'min' (trim x axis to minimum number of available episodes among experiments)")

    return parser.parse_args()

class ResultFolder:
    def __init__(self, path, parameter_name='phi_values'):
        self._folder_path = path
        self._results = None
        self._max_episode = 0
        self._parameter_name = parameter_name
        self._parse_folder()

    def _parse_file(self, file_obj, start):
        data_list = json.load(file_obj)
        if type(data_list) != dict:
            if self._results is None:
                self._results = defaultdict(lambda: [])
            episode_index = start
            for episode_data in data_list:
                self._results[episode_index] = episode_data
                episode_index += 1
            if episode_index > self._max_episode:
                self._max_episode = episode_index
        else:
            if self._results is None:
                self._results = defaultdict(lambda: defaultdict(lambda: []))
            for subparam, subparam_data_list in data_list.items():
                episode_index = start
                for episode_data in subparam_data_list:
                    self._results[subparam][episode_index] = episode_data
                    episode_index += 1
                if episode_index > self._max_episode:
                    self._max_episode = episode_index

    def _parse_folder(self, path=None):
        cur_path = path if path else self._folder_path
        with os.scandir(cur_path) as files:
            for file in files:
                if file.is_dir(follow_symlinks=False):
                    self._parse_folder(file.path)
                    continue
                if not file.is_file():
                    continue
                match = re.search(f'({self._parameter_name})_([0-9]+)-([0-9]+).json', file.name)
                if not match:
                    continue
                with open(file.path, 'r') as f:
                    self._parse_file(f, int(match[2]))
                # print(f'parsed file {file.name}')
        print(f'parsed folder {cur_path}')

def get_averaged_values(values, plot_period):
    averaged_values = []
    number_of_points = (len(values) // plot_period)
    if len(values) % plot_period != 0:
        number_of_points += 1

    for i in range(number_of_points):
        start = i * plot_period
        end = min((i + 1) * plot_period, len(values))
        averaged_values.append(sum(values[start:end]) / (end - start))

    return averaged_values

def get_minmax_values(values, plot_period):
    min_values = []
    max_values = []
    number_of_points = (len(values) // plot_period)
    if len(values) % plot_period != 0:
        number_of_points += 1

    for i in range(number_of_points):
        start = i * plot_period
        end = min((i + 1) * plot_period, len(values))
        min_values.append(min(values[start:end]))
        max_values.append(max(values[start:end]))

    return min_values, max_values

def rolling_average(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def get_best_phi(phi_values, averaging_period):
    return np.min(rolling_average(phi_values, averaging_period))

def get_config_value(exp_dir, key):
    filename = os.path.join(exp_dir, "config.yaml")
    key = key.strip()
    if key.endswith(":"):
        key = key[:-1]
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(key):
                val = line[len(key):].strip()
                if val.startswith(":"):
                    val = val[1:].strip()
                    return val
                else:
                    continue
    return None

def plot_advanced_avg(result_paths, exp_names, values, title, out_filename,
    first_values=None, update_period={}, df_genetic=None, ylim=(None, None), n_episodes='max',
    plot_period=None, best_phi_avg_period=2000):
    title_fontsize = 10
    legend_fontsize = 9

    plt.style.use(["science", "grid"])
    matplotlib.rcParams['text.usetex'] = False

    markers = ['x', '.', 'v', '^', 'd']
    colors = (matplotlib.rcParams['axes.prop_cycle'].by_key()['color'] * 4)[:len(markers)]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(160/23, 4), dpi=300)

    phi_ga = None
    if df_genetic is not None:
        try:
            df_on_train = df_genetic[(df_genetic.hash_funcs_set == "train") &
                                     (df_genetic.iteration >= 0)]
            phi_ga = df_on_train[df_on_train["iteration"] == df_on_train["iteration"].max()]["phi_min"].iloc[0]
        except Exception as e:
            print(f"ERROR plotting genetic algorithm result: {e}")

    if n_episodes == 'max':
        n_episodes_display = max([len(values[i][params[0]]) for i in range(len(exp_names))])
    elif n_episodes == 'min':
        n_episodes_display = min([len(values[i][params[0]]) for i in range(len(exp_names))])
    else:
        n_episodes_display = n_episodes

    if plot_period is None:
        plot_period = ceil(n_episodes_display / 5)

    n_episodes_xlim = round(n_episodes_display*1.28/500)*500
    if n_episodes_display < best_phi_avg_period:
        best_phi_avg_period = n_episodes_display // 5

    phi_best_values = []
    for i, exp_name in enumerate(exp_names):
        y_orig = np.array(values[i][params[0]])
        y_orig = y_orig[:n_episodes_display]
        phi_best = get_best_phi(y_orig, best_phi_avg_period)
        phi_best_values.append(phi_best)
        val1 = np.array(values[i]['message_iterations_done'])
        val2 = np.array(values[i]['message_iterations_possible'])
        upd_n = update_period[i]
        if np.sum(val2) != 0:
            val1d = np.concatenate(([val1[0]], val1[1:] - val1[:-1]))
            val2d = np.concatenate(([val2[0]], val2[1:] - val2[:-1]))
            val1d = np.array([np.sum(val1d[j*upd_n:(j+1)*upd_n]) for j in range(len(val1d)//upd_n)])
            val2d = np.array([np.sum(val2d[j*upd_n:(j+1)*upd_n]) for j in range(len(val2d)//upd_n)])
            msg_ratio = val1d / val2d
        else:
            msg_ratio = np.ones(len(val1)//upd_n).astype('float')
        avg_msg_load = np.mean(msg_ratio)
        if avg_msg_load == 1.0:
            avg_msg_load_str = ""
        else:
            avg_msg_load_str = f": share of remaining exchanges = {avg_msg_load*100:.1f}%"

        y = np.array(get_averaged_values(y_orig, plot_period))
        x = np.arange(0, plot_period*(len(y)+2), plot_period)[1:len(y)]
        x = np.concatenate((x, [len(y_orig)]))
        ax.plot(x, y, marker=markers[i], color=colors[i], linewidth=1.0,
                label=f"{exp_name}{avg_msg_load_str}")

    ax.set_xlim(0, n_episodes_xlim)
    xlim = ax.get_xlim()
    ylim_cur = ax.get_ylim()
    height_cur = ylim_cur[1] - ylim_cur[0]

    if first_values is not None:
        phi_eqw = first_values[0]["phi_values"][0]
        ax.hlines(phi_eqw, xlim[0], xlim[1], color='gray', linestyle='--', label="ECMP")
        ax.text(n_episodes_display*0.04, phi_eqw + height_cur*0.015, f"{phi_eqw:.4f}", fontsize=8)
    if phi_ga is not None:
        ax.hlines(phi_ga, xlim[0], xlim[1], color='darkolivegreen', linestyle='-', label="centralized genetic algorithm")
        ax.text(n_episodes_display*0.04, phi_ga + height_cur*0.015, f"{phi_ga:.4f}", fontsize=8)

    if ylim != (None, None):
        ax.set_ylim(*ylim)
    else:
        # correcting ylims so that the legend likely won't overlap the plot
        ylim_cur_0, ylim_cur_1 = ax.get_ylim()
        ylim_cur_1 = ylim_cur_0 + (ylim_cur_1 - ylim_cur_0) * 1.5
        ax.set_ylim([ylim_cur_0, ylim_cur_1])
    ylim_cur = ax.get_ylim()
    height_cur = ylim_cur[1] - ylim_cur[0]

    phi_best_values = np.array(phi_best_values)
    phi_best_argsort = np.argsort(-phi_best_values)
    phi_best_values_sort = phi_best_values[phi_best_argsort]
    positions = []
    cur_pos = -1
    text_y_thres = 0.044
    for i, phi_best in enumerate(phi_best_values_sort):
        if i > 0 and (phi_best - phi_best_values_sort[i-1]) / height_cur < text_y_thres:
            cur_pos = -cur_pos
        positions.append(cur_pos)
    positions = np.array(positions)
    positions = positions[phi_best_argsort.argsort()]

    for i, (exp_name, phi_best) in enumerate(zip(exp_names, phi_best_values)):
        ax.scatter(n_episodes_xlim*0.91 - n_episodes_xlim*0.02*(positions[i]<0),
            phi_best, color=colors[i], marker=markers[i])
        ax.text(n_episodes_xlim*(0.91-0.045) + n_episodes_xlim*0.055*positions[i],
            phi_best, f"{phi_best:.4f}", fontsize=8)
    ax.text(n_episodes_xlim*(0.91-0.045) - n_episodes_xlim*0.055,
            max(phi_best_values) + height_cur * 0.1, "Best averaged:", fontsize=9)

    title_full = f"{title} (display batch = {plot_period} episodes)"
    ax.set_title(title_full, fontsize=title_fontsize)
    ax.set_xlabel("episode")
    ax.set_ylabel("Ф value")

    xticks = set(range(0, n_episodes_display, plot_period))
    xticks.add(n_episodes_display)
    ax.set_xticks(sorted(xticks))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1000))

    ax.legend(fontsize=legend_fontsize, labelspacing=0.23, loc='upper right')
    plt.savefig(out_filename, bbox_inches='tight')
    print(f"plot with average value lines saved to {out_filename}")
    plt.close()

def plot_advanced_avgminmax(result_paths, exp_names, values, title, out_filename,
    first_values=None, update_period={}, df_genetic=None, ylim=(None, None), n_episodes='max',
    plot_period=None, best_phi_avg_period=2000, write_exchanges_percent=False):
    title_fontsize = 10
    legend_fontsize = 9

    plt.style.use(["science", "grid"])
    matplotlib.rcParams['text.usetex'] = False

    markers = ['x', '.', 'v', '^', 'd']
    colors = (matplotlib.rcParams['axes.prop_cycle'].by_key()['color'] * 4)[:len(markers)]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(160/23, 4), dpi=300)

    phi_ga = None
    if df_genetic is not None:
        try:
            df_on_train = df_genetic[(df_genetic.hash_funcs_set == "train") &
                                     (df_genetic.iteration >= 0)]
            phi_ga = df_on_train[df_on_train["iteration"] == df_on_train["iteration"].max()]["phi_min"].iloc[0]
        except Exception as e:
            print(f"ERROR plotting genetic algorithm result: {e}")

    if n_episodes == 'max':
        n_episodes_display = max([len(values[i][params[0]]) for i in range(len(exp_names))])
    elif n_episodes == 'min':
        n_episodes_display = min([len(values[i][params[0]]) for i in range(len(exp_names))])
    else:
        n_episodes_display = n_episodes

    if plot_period is None:
        plot_period = ceil(n_episodes_display / 5)

    n_episodes_xlim = round(n_episodes_display*1.28/500)*500
    if n_episodes_display < best_phi_avg_period:
        best_phi_avg_period = n_episodes_display // 5

    figsize_height_fix = 0.865
    fig, axes = plt.subplots(nrows=1, ncols=len(exp_names),
                             figsize=(160/23, 4*figsize_height_fix),
                             sharey=True, dpi=300)
    plt.subplots_adjust(wspace=0.1)

    if first_values is not None:
        phi_eqw = first_values[0]["phi_values"][0]

    ylim_cur = axes[0].get_ylim()
    height_cur = ylim_cur[1] - ylim_cur[0]
    for ax in axes:
        ax.set_xlim(0, n_episodes_xlim)
        xlim = ax.get_xlim()
        if first_values is not None:
            ax.hlines(phi_eqw, xlim[0], xlim[1], color='gray',
                      linestyle='--', linewidth=1.1, label="ECMP")
        if phi_ga is not None:
            ax.hlines(phi_ga, xlim[0], xlim[1], color='darkolivegreen',
                      linestyle='-', linewidth=0.8, label="centralized genetic algorithm")

    phi_best_values = []
    xticks = []
    title_full = f"{title} (display batch = {plot_period} episodes)"
    for i, exp_name in enumerate(exp_names):
        y_orig = np.array(values[i][params[0]])
        y_orig = y_orig[:n_episodes_display]
        phi_best = get_best_phi(y_orig, best_phi_avg_period)
        phi_best_values.append(phi_best)

        if write_exchanges_percent:
            val1 = np.array(values[i]['message_iterations_done'])
            val2 = np.array(values[i]['message_iterations_possible'])
            upd_n = update_period[i]
            if np.sum(val2) != 0:
                val1d = np.concatenate(([val1[0]], val1[1:] - val1[:-1]))
                val2d = np.concatenate(([val2[0]], val2[1:] - val2[:-1]))
                val1d = np.array([np.sum(val1d[j*upd_n:(j+1)*upd_n]) for j in range(len(val1d)//upd_n)])
                val2d = np.array([np.sum(val2d[j*upd_n:(j+1)*upd_n]) for j in range(len(val2d)//upd_n)])
                msg_ratio = val1d / val2d
            else:
                msg_ratio = np.ones(len(val1)//upd_n).astype('float')
            avg_msg_load = np.mean(msg_ratio)
            if avg_msg_load == 1.0:
                avg_msg_load_str = ""
            else:
                avg_msg_load_str = f": share of remaining exchanges = {avg_msg_load*100:.1f}%"
        else:
            avg_msg_load_str = ""

        y = np.array(get_averaged_values(y_orig, plot_period))
        ymin, ymax = get_minmax_values(y_orig, plot_period)
        ymin = np.array(ymin)
        ymax = np.array(ymax)
        x = np.arange(0, plot_period*(len(y)+2), plot_period)[1:len(y)]
        x = np.concatenate((x, [len(y_orig)]))
        xticks.append(x)
        axes[i].errorbar(x, y, yerr=np.vstack((y - ymin, ymax - y)),
                         marker=markers[i], color=colors[i],
                         capsize=3, linewidth=1.33,
                         markeredgewidth=1.2,
                         label=f"{exp_name}{avg_msg_load_str}")

        ax_title = textwrap.fill(f"{exp_name}{avg_msg_load_str}", 20, break_long_words=False)
        axes[i].set_title(ax_title, fontsize=title_fontsize)
        axes[i].set_xlabel("episode")
        xticks_sort = np.sort(x)
        axes[i].set_xticks(xticks_sort)
        axes[i].xaxis.set_minor_locator(plt.MultipleLocator(1000))
        if np.min(xticks_sort[1:] - xticks_sort[:-1]) >= plot_period * 0.85:
            rotation = 30
        else:
            rotation = 60
        axes[i].tick_params(axis='x', labelsize=8, rotation=rotation)

    ylim_cur = axes[0].get_ylim()
    height_cur = ylim_cur[1] - ylim_cur[0]

    if ylim != (None, None):
        axes[0].set_ylim(*ylim)
    ylim_cur = axes[0].get_ylim()
    height_cur = ylim_cur[1] - ylim_cur[0]

    if first_values is not None:
        axes[0].text(xlim[1] * (1-0.02), phi_eqw + height_cur * 0.015, "ECMP",
                horizontalalignment='right', fontsize=7)
    if phi_ga is not None:
        axes[0].text(xlim[1] * (1-0.02), phi_ga + height_cur * 0.015, "gen.\nalg.",
                horizontalalignment='right', fontsize=7)

    title_full = f"{title} (display batch = {plot_period} episodes)"
    axes[0].set_ylabel("Ф value")

    fig.suptitle(title_full, y=1.15 if write_exchanges_percent else 1.1,
                 fontsize=title_fontsize)
    plt.savefig(out_filename, bbox_inches='tight')
    print(f"plot with min-max ranges saved to {out_filename}")
    plt.close()

if __name__ == "__main__":
    args = parse_args()

    with open(args.experiments_list, "r") as f:
        exp_dict = json.load(f)
    title = exp_dict.get("title", "")
    experiment_dicts = exp_dict["experiments"]
    genetic_filename = exp_dict.get("genetic", None)
    result_paths = []
    exp_names = []
    for d in experiment_dicts:
        path = d["path"]
        name = d.get("name")
        if name is None:
            name = path
        result_paths.append(path)
        exp_names.append(name)

    n_episodes = args.episodes
    if n_episodes not in ['min', 'max']:
        n_episodes = int(n_episodes)

    params = ["phi_values", "messages"]
    folders = defaultdict(dict)
    values = defaultdict(dict)
    first_values = defaultdict(dict)
    update_period = {}
    for i, result_path in enumerate(result_paths):
        update_period[i] = 1
        config_path = os.path.join(result_path, "config.yaml")
        try:
            with open(config_path, "r") as f:
                for line in f:
                    m = re.match(r"n_without_update: (\d+)", line.strip())
                    if m is not None:
                        update_period[i] = int(m.groups()[0])
        except Exception as e:
            print(f"ERROR reading {config_path} for n_without_update value: {e}")

        for param in params:
            folders[i][param] = ResultFolder(result_path, parameter_name=param)
            res = folders[i][param]._results
            if res is None:
                print(result_path)
            if param == "phi_values":
                max_episode = len(res)
                values[i][param] = [res[episode][-1] for episode in range(0, max_episode)]
                first_values[i][param] = [res[episode][0] for episode in range(0, max_episode)]
            elif param == "messages":
                for subparam, res_subparam in res.items():
                    max_episode = len(res_subparam)
                    values[i][subparam] = [res_subparam[episode] for episode in range(0, max_episode)]
            elif param in ["messages_infer", "messages_train"]:
                for subparam, res_subparam in res.items():
                    max_episode = len(res_subparam)
                    values[i][subparam] = [res_subparam[episode] for episode in range(0, max_episode)]
            else:
                values[i][param] = [res[episode] for episode in range(0, max_episode)]

    # phi0 = first_values[0]["phi_values"][0]
    # for episode in range(len(first_values)):
    #     for x in first_values[episode]["phi_values"]:
    #         assert x == phi0

    if genetic_filename is not None:
        df_genetic = pd.read_csv(genetic_filename)
    else:
        df_genetic = None

    ymin = args.ymin
    ymax = args.ymax
    dat_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')[:-3]
    out_filename_avg = f"{args.experiments_list.stem}_avg_{dat_str}.png"
    out_filename_avgminmax = f"{args.experiments_list.stem}_avgminmax_{dat_str}.png"
    plot_advanced_avg(result_paths, exp_names, values, title, out_filename=out_filename_avg, first_values=first_values,
        update_period=update_period, df_genetic=df_genetic, n_episodes=n_episodes, ylim=(ymin, ymax))
    plot_advanced_avgminmax(result_paths, exp_names, values, title, out_filename=out_filename_avgminmax, first_values=first_values,
        update_period=update_period, df_genetic=df_genetic, n_episodes=n_episodes, ylim=(ymin, ymax),
        write_exchanges_percent=False)
