import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scienceplots
import json
from collections import Counter, defaultdict
import os
import re
import glob
import pandas as pd
import pathlib
from pathlib import Path
from datetime import datetime

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

# plt.rcParams.update({'font.size': 9})

class ResultFolder:
    def __init__(self, path, parameter_name='phi_values'):
        self._folder_path = path
        # self._results = defaultdict(lambda: [])
        self._results = None
        self._results2 = []
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

    def _parse_file2(self, file_obj):
        data_list = json.load(file_obj)
        for run_data in data_list:
            self._results2.append(run_data[0])

    def _parse_folder2(self, path=None):
        cur_path = path if path else self._folder_path
        with os.scandir(cur_path) as files:
            for file in files:
                if not file.is_file():
                    continue
                match = re.search('avg.json', file.name)
                if not match:
                    continue
                with open(file.path, 'r') as f:
                    self._parse_file2(f)
                print(f'parsed file {file.name}')

    def _plot(self, values: list, start: int, end: int, amount: int, x_name: str, filename: str):
        plt.rcParams.update({'font.size': 22})
        # plt.rc('xtick', labelsize=14)

        np_phi_values = np.array(values)
        np_linear = np.linspace(start, end, amount)
        pixel = 1/plt.rcParams['figure.dpi']
        figure, ax = plt.subplots(figsize=(1200*pixel, 800*pixel))
        ax.plot(np_linear, np_phi_values, label='phi')
        ax.set_xlabel(x_name)
        ax.set_ylabel(self._parameter_name)
        # ax.legend()
        # plt.ylim(0.01, 0.25)
        # plt.yticks([0.02, 0.022, 0.024, 0.026, ])
        plt.savefig(os.path.join(self._folder_path, filename), bbox_inches='tight')
        plt.close()

    def _boxplot(self, values: list, episodes_per_matrix: int, x_name: str, filename: str):
        box_values = []
        for matrix_num in range(0, len(values) // episodes_per_matrix):
            box_values.append(np.array(values[matrix_num * episodes_per_matrix:(matrix_num + 1) * episodes_per_matrix]))
        np_phi_values = np.transpose(np.array(box_values))
        pixel = 1/plt.rcParams['figure.dpi']
        figure, ax = plt.subplots(figsize=(1900*pixel, 800*pixel))
        ax.boxplot(np_phi_values)
        # ax.set_xlabel(x_name)
        # ax.set_ylabel('phi value')
        # ax.legend()
        plt.savefig(os.path.join(self._folder_path, filename), bbox_inches='tight')
        plt.close()

    def prepare_phi_values(self, max_episode=None):
        if max_episode:
            self._max_episode = max_episode

        # phi_values = [min(self._results[episode]) for episode in range(max_episode)]
        phi_values = [self._results[episode][-1] for episode in range(self._max_episode)]

        return phi_values

    def prepare_averaged_phi_values(self):
        # max_episode = min(self._max_episode, 1000)
        # phi_values = [min(self._results[episode]) for episode in range(max_episode)]
        plot_period = 5

        phi_values = [self._results[episode][-1] for episode in range(self._max_episode)]

        averaged_values = []
        number_of_points = (len(phi_values) // plot_period)
        if len(phi_values) % plot_period != 0:
            number_of_points += 1

        for i in range(number_of_points):
            start = i * plot_period
            end = min((i + 1) * plot_period, len(phi_values))
            averaged_values.append(sum(phi_values[start:end]) / (end - start))

        return averaged_values

    def plot_average_and_full(self, plot_period, parameter_name='phi_values', max_episode=None):
        if max_episode:
            self._max_episode = max_episode

        # phi_values = [min(self._results[episode]) for episode in range(self._max_episode)]
        phi_values = [self._results[episode][-1] for episode in range(self._max_episode)]

        averaged_values = []
        number_of_points = (len(phi_values) // plot_period)
        if len(phi_values) % plot_period != 0:
            number_of_points += 1

        for i in range(number_of_points):
            start = i * plot_period
            end = min((i + 1) * plot_period, len(phi_values))
            averaged_values.append(sum(phi_values[start:end]) / (end - start))

        # full graph
        self._plot(phi_values, 0, len(phi_values) - 1, len(phi_values), 'episode', f'{self._parameter_name}_0-{len(phi_values)}.png')

        # averaged graph
        self._plot(averaged_values, 0, len(averaged_values) - 1, len(averaged_values),
                   f'episode batch ({plot_period} per batch)', f'{self._parameter_name}_averaged_full.png')

def get_averaged_values(values, plot_period=5):
    averaged_values = []
    number_of_points = (len(values) // plot_period)
    if len(values) % plot_period != 0:
        number_of_points += 1

    for i in range(number_of_points):
        start = i * plot_period
        end = min((i + 1) * plot_period, len(values))
        averaged_values.append(sum(values[start:end]) / (end - start))

    return averaged_values

def get_minmax_values(values, plot_period=5):
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

def plot_advanced(result_paths, exp_names, values, title, out_filename,
    first_values=None, update_period={}, df_genetic=None, ylim=(None, None), n_episodes='max',
    plot_period=3000, best_phi_avg_period=2000):
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
    n_episodes_xlim = round(n_episodes_display*1.28/500)*500
    if n_episodes_display < best_phi_avg_period:
        best_phi_avg_period = n_episodes_display // 5

    phi_best_values = []
    for i, exp_name in enumerate(exp_names):
        title = f"{exp_name}"
        y_orig = np.array(values[i][params[0]])
        y_orig = y_orig[:n_episodes_display]
        xshift = (plot_period * i / len(exp_names) - plot_period * (len(exp_names)-1) / len(exp_names)) * 0.3
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
            avg_msg_load_str = f" (comm: {avg_msg_load*100:.1f}%)"

        y = np.array(get_averaged_values(y_orig, plot_period))
        ymin, ymax = get_minmax_values(y_orig, plot_period)
        ymin = np.array(ymin)
        ymax = np.array(ymax)
        x = np.arange(0, plot_period*(len(y)+2), plot_period)[1:len(y)+1]
        alpha = 0.1 if len(exp_names) <= 2 else 0.07
        ax.errorbar(x+xshift, y, yerr=np.vstack((y - ymin, ymax - y)),
                    marker=markers[i], color=colors[i],
                    linewidth=1.33, label=f"{exp_name}{avg_msg_load_str}")
        ax.plot(x+xshift, ymin, linestyle='dotted', color=colors[i], marker=markers[i], ms=5)
        ax.plot(x+xshift, ymax, linestyle='dotted', color=colors[i], marker=markers[i], ms=5)

    ax.set_xlim(0, n_episodes_xlim)
    xlim = ax.get_xlim()
    ylim_cur = ax.get_ylim()
    height_cur = ylim_cur[1] - ylim_cur[0]

    if first_values is not None:
        phi_eqw = first_values[0]["phi_values"][0]
        ax.hlines(phi_eqw, xlim[0], xlim[1], color='gray', linestyle='--', label="equal weights")
        ax.scatter(n_episodes_display*0.05, phi_eqw, marker="*", color='gray')
        ax.text(n_episodes_display*0.06, phi_eqw + height_cur*0.01, f"{phi_eqw:.4f}", fontsize=8)
    if phi_ga is not None:
        ax.hlines(phi_ga, xlim[0], xlim[1], color='darkolivegreen', linestyle='-', label="genetic algorithm")
        ax.scatter(n_episodes_display*0.05, phi_ga, marker="*", color='darkolivegreen')
        ax.text(n_episodes_display*0.06, phi_ga + height_cur*0.01, f"{phi_ga:.4f}", fontsize=8)

    if ylim != (None, None):
        ax.set_ylim(*ylim)
    ylim_cur = ax.get_ylim()

    phi_best_values = np.array(phi_best_values)
    phi_best_argsort = np.argsort(phi_best_values)
    phi_best_values_sort = phi_best_values[phi_best_argsort]
    positions = []
    cur_pos = 1
    text_y_thres = 0.024
    for i, phi_best in enumerate(phi_best_values_sort):
        if i > 0 and (phi_best - phi_best_values_sort[i-1]) / (ylim_cur[1] - ylim_cur[0]) < text_y_thres:
            cur_pos = -cur_pos
        positions.append(cur_pos)
    positions = np.array(positions)
    positions = positions[phi_best_argsort.argsort()]

    for i, (exp_name, phi_best) in enumerate(zip(exp_names, phi_best_values)):
        ax.scatter(n_episodes_xlim*0.91 - n_episodes_xlim*0.02*(positions[i]<0),
            phi_best, color=colors[i], marker=markers[i])
        ax.text(n_episodes_xlim*(0.91-0.045) + n_episodes_xlim*0.055*positions[i],
            phi_best, f"{phi_best:.4f}", fontsize=8)

    title_full = f"{title} (display batch = {plot_period} episodes)"
    ax.set_title(title_full, fontsize=title_fontsize)
    ax.set_xlabel("episode")
    ax.set_ylabel("Ф value")

    ax.legend(fontsize=legend_fontsize, labelspacing=0.23, loc='upper right')
    plt.savefig(out_filename, bbox_inches='tight')
    print(f"plot saved to {out_filename}")
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
    # values_averaged = defaultdict(dict)
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
                # values_averaged[i][param] = get_averaged_values(values[i][param])
            elif param == "messages":
                for subparam, res_subparam in res.items():
                    max_episode = len(res_subparam)
                    values[i][subparam] = [res_subparam[episode] for episode in range(0, max_episode)]
                    # values_averaged[i][subparam] = get_averaged_values(values[i][subparam])
            elif param in ["messages_infer", "messages_train"]:
                for subparam, res_subparam in res.items():
                    max_episode = len(res_subparam)
                    values[i][subparam] = [res_subparam[episode] for episode in range(0, max_episode)]
                    # values_averaged[i][subparam] = get_averaged_values(values[i][subparam])
            else:
                values[i][param] = [res[episode] for episode in range(0, max_episode)]
                # values_averaged[i][param] = get_averaged_values(values[i][param])

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
    out_filename = f"{args.experiments_list.stem}_{dat_str}.png"
    plot_advanced(result_paths, exp_names, values, title, out_filename=out_filename, first_values=first_values,
        update_period=update_period, df_genetic=df_genetic, n_episodes=n_episodes, ylim=(ymin, ymax))
