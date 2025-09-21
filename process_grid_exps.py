import numpy as np
import argparse
import json
from collections import defaultdict
import os
import re
import glob
import pandas as pd

import pathlib
from pathlib import Path

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
        print(f'parsed folder {cur_path}')

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Parse results of experiment series (see Readme) to reproduce Tables 2-5 from the paper.'
    )
    parser.add_argument(
        'series_name',
        type=str,
        help='Name of experiment series. '
             'Choices: "abilene_traj" (Table 2 in paper), "rhombus" (Table 3), "abilene" (Tables 4-5)'
    )
    args = parser.parse_args()

    grid_name = args.series_name
    grids_rootdir = "data_examples/grids"

    link_state_size = 16
    n_epochs = 3
    float_bytes = 4
    if grid_name == "rhombus":
        folder_name = "rhombus_grid"
        exp_prefix = "rh"
        ep_crop = 6000
        n_episodes_full = 6000
        n_horizons_maroh = 100
        n_horizons_samaroh = 20
        n_msg_iter = 2
        sum_square_degrees = 16

        config_list = \
            [("MAROH", 0, "-", "-")] + \
            [("SAMAROH", 0, "-", "-")] + \
            [("MAROH-2L", mem_size, "l2", thres) for mem_size in ["512"] for thres in ['0.007', '0.010', '0.015', '0.018', '0.025', '0.030', '0.040', '0.050', '0.060', '0.070']] + \
            [("SAMAROH-2L", mem_size, "l2", thres) for mem_size in ["512"] for thres in ['0.007', '0.010', '0.015', '0.018']] + \
            [("SAMAROH-2L", mem_size, "l1", thres) for mem_size in ["512"] for thres in ['0.01', '0.03', '0.04', '0.05']] + \
            [("SAMAROH-2L", mem_size, "cosine", thres) for mem_size in ["512"] for thres in ['0.0000001', '0.0000003', '0.0000004', '0.0000005']]

    elif grid_name == "abilene":
        folder_name = "abilene_grid"
        exp_prefix = "ab"
        ep_crop = 15000
        n_episodes_full = 15000
        n_horizons_maroh = 250
        n_horizons_samaroh = 50
        n_msg_iter = 3
        sum_square_degrees = 60

        config_list = \
            [("MAROH", 0, "-", "-")] + \
            [("SAMAROH", 0, "-", "-")] + \
            [("SAMAROH-2L", mem_size, "l2", thres) for mem_size in ["512", "1024"] for thres in ['0.035', '0.04', '0.05', '0.06']] + \
            [("SAMAROH-2L", mem_size, "l1", thres) for mem_size in ["512", "1024"] for thres in ['0.052', '0.062', '0.067', '0.077']] + \
            [("SAMAROH-2L", mem_size, "cosine", thres) for mem_size in ["512", "1024"] for thres in ['0.00000015', '0.0000003', '0.00000045', '0.00000075']]

    elif grid_name == "abilene_traj":
        folder_name = "abilene_traj_grid"
        exp_prefix = "ab"
        ep_crop = 8000
        n_episodes_full = 8000
        n_horizons_maroh = 250
        n_horizons_samaroh = 50
        n_msg_iter = 3
        sum_square_degrees = 60

        config_list = [(method, update_period, lr_factor) for method in ["SAMAROH", "MAROH"]
             for update_period in [1, 5, 25, 75] for lr_factor in [1, 5]]

    else:
        raise ValueError("invalid name of experiment series")

    # find "exp_*" folders (results of experiment runs) inside subfolders of experiment series folder

    task_subdirs = []
    task_subdir_params = []
    result_paths_zip = []

    if grid_name in ["abilene", "rhombus"]:
        for (method, mem_size, metric_alias, threshold) in config_list:
            if metric_alias == "l2-aggl":
                clustering = "Agglomerative"
                metric = "l2"
            elif metric_alias == "l2":
                clustering = "MiniBatchKMeans"
                metric = "l2"
            else:
                clustering = "Agglomerative"
                metric = metric_alias
            if method.upper().startswith("SAMAROH"):
                method_str = ""
            elif method.upper().startswith("MAROH"):
                method_str = "_nonsa"
            else:
                raise ValueError("invalid method name")
            if "2L" not in method.upper():
                task_dir_res = os.path.join(folder_name, f"{exp_prefix}{method_str}_nomem")
            else:
                task_dir_res = os.path.join(folder_name, f"{exp_prefix}{method_str}_{mem_size}_{metric_alias}_{threshold}")
            for subdir in glob.glob(os.path.join(grids_rootdir, task_dir_res, 'exp_*')):
                task_subdirs.append(subdir)
                task_subdir_params.append((method, mem_size, metric_alias, clustering, metric, threshold))

        for subdir, (method, mem_size, metric_alias, clustering, metric, threshold) in zip(task_subdirs, task_subdir_params):
            result_paths_zip.append((subdir, f"{method}, mem_size={mem_size}, metric={metric_alias}, thres={threshold}"))

    elif grid_name in ["abilene_traj"]:
        for (method, update_period, lr_factor) in config_list:
            if method.upper().startswith("SAMAROH"):
                method_str = ""
            elif method.upper().startswith("MAROH"):
                method_str = "_nonsa"
            else:
                raise ValueError("invalid method name")
            task_dir_res = os.path.join(folder_name, f"{exp_prefix}{method_str}_upd{update_period}_lrx{lr_factor}")
            for subdir in glob.glob(os.path.join(grids_rootdir, task_dir_res, 'exp_*')):
                task_subdirs.append(subdir)
                task_subdir_params.append((method, update_period, lr_factor))

        for subdir, (method, update_period, lr_factor) in zip(task_subdirs, task_subdir_params):
            result_paths_zip.append((subdir, f"{method}\t{update_period}\tlr*{lr_factor}"))

    # parse values from json files in found "exp_*" folders

    result_paths, exp_names = list(zip(*result_paths_zip))
    result_paths = list(result_paths)
    exp_names = list(exp_names)

    params = ["phi_values", "messages"]
    folders = defaultdict(dict)
    values = defaultdict(dict)
    update_period_dict = {}
    for i, result_path in enumerate(result_paths):
        update_period_dict[i] = 1
        config_path = os.path.join(result_path, "config.yaml")
        try:
            with open(config_path, "r") as f:
                for line in f:
                    m = re.match(r"n_without_update: (\d+)", line.strip())
                    if m is not None:
                        update_period_dict[i] = int(m.groups()[0])
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
            elif param == "messages":
                for subparam, res_subparam in res.items():
                    max_episode = len(res_subparam)
                    values[i][subparam] = [res_subparam[episode] for episode in range(0, max_episode)]
            elif param == "messages_infer":
                for subparam, res_subparam in res.items():
                    max_episode = len(res_subparam)
                    values[i][subparam] = [res_subparam[episode] for episode in range(0, max_episode)]
            elif param == "messages_train":
                for subparam, res_subparam in res.items():
                    max_episode = len(res_subparam)
                    values[i][subparam] = [res_subparam[episode] for episode in range(0, max_episode)]
            else:
                values[i][param] = [res[episode] for episode in range(0, max_episode)]

    # aggregate phi and number of exchanges data and per experimental run

    last_period = 2000
    results = []

    for i, (path, exp_name, params_tuple) in enumerate(zip(result_paths, exp_names, task_subdir_params)):
        upd_n = update_period_dict[i]
        if grid_name in ["abilene", "rhombus"]:
            method, mem_size, metric_alias, clustering, metric, threshold = params_tuple
        elif grid_name in ["abilene_traj"]:
            method, update_period, lr_factor = params_tuple
            assert upd_n == update_period

        result_path = Path(path)

        result_path_short = Path(*Path(result_path).parts[-2:])
        try:
            y_orig = np.array(values[i][params[0]])
        except KeyError:
            continue
        if len(y_orig) < last_period:
            continue

        y_orig = y_orig[:ep_crop]
        n_episodes = len(y_orig)
        if n_episodes < n_episodes_full:
            progress_str = f"({n_episodes}/{n_episodes_full} ep)"
        else:
            progress_str = ""
        last_phi = np.mean(y_orig[-last_period:])

        val1 = np.array(values[i]['message_iterations_done'])[:ep_crop]
        val2 = np.array(values[i]['message_iterations_possible'])[:ep_crop]

        if np.sum(val2) != 0:
            val1d = np.concatenate(([val1[0]], val1[1:] - val1[:-1]))
            val2d = np.concatenate(([val2[0]], val2[1:] - val2[:-1]))
            val1d = np.array([np.sum(val1d[j*upd_n:(j+1)*upd_n]) for j in range(len(val1d)//upd_n)])
            val2d = np.array([np.sum(val2d[j*upd_n:(j+1)*upd_n]) for j in range(len(val2d)//upd_n)])
            msg_ratio = val1d / val2d
        else:
            msg_ratio = np.ones(len(val1)//upd_n).astype('float')
        avg_msg_load = np.mean(msg_ratio)
        if method.upper().startswith("MAROH"):
            n_horizons = n_horizons_maroh
        elif method.upper().startswith("SAMAROH"):
            n_horizons = n_horizons_samaroh
        else:
            raise ValueError("invalid method name")
        bytes_per_ep_possible = n_horizons * n_msg_iter * sum_square_degrees * (1 + n_epochs) * link_state_size * float_bytes
        mbytes_per_ep = avg_msg_load * bytes_per_ep_possible / 1024 / 1024
        print(f"{exp_name:60}\t{last_phi:.4f}\t{avg_msg_load*100:.2f}%\t{mbytes_per_ep:.3f} MB/ep\t{progress_str}\t{result_path_short}")
        if n_episodes == n_episodes_full:
            if grid_name in ["abilene", "rhombus"]:
                results.append((method, mem_size, clustering, metric, threshold, last_phi, avg_msg_load*100, mbytes_per_ep))
            elif grid_name in ["abilene_traj"]:
                lrs = f"[0.00003 * {lr_factor}, 0.00017 * {lr_factor}]"
                results.append((method, update_period, lrs, last_phi, avg_msg_load*100, mbytes_per_ep))

    if grid_name in ["abilene", "rhombus"]:
        agg_columns = ["method", "memory size", "clustering", "metric", "threshold"]
    elif grid_name in ["abilene_traj"]:
        agg_columns = ["method", "trajectory length", "optimizer step sizes"]
    columns = agg_columns + ["phi", "number of exchanges, %", "data exchanged per episode, MB"]
    df = pd.DataFrame(results, columns=columns)

    # aggregate data per same configuration of parameters (with multiple experimental runs per each configuration)

    new_columns = agg_columns + [f'{col}{postfix}' for col in df.columns if col not in agg_columns
        for postfix in [' (mean)', ' (std)']]
    df_stats = pd.concat([
        df.groupby(agg_columns, sort=False, as_index=False).mean().rename(
            columns={c: c+' (mean)' for c in df.columns}),
        df.groupby(agg_columns, sort=False, as_index=False).std(ddof=1).rename(
            columns={c: c+' (std)' for c in df.columns if c not in agg_columns})
    ], axis=1)[new_columns]

    print("stats table:")
    print(df_stats)

    filename_verbose = f'{grid_name}_table_verbose.csv'
    filename_stats = f'{grid_name}_table_stats.csv'

    df.to_csv(filename_verbose)
    print(f"verbose table (individual values for each seed) written to: {filename_verbose}")
    df_stats.to_csv(filename_stats)
    print(f"stats table (mean, std per parameters configuration) written to: {filename_stats}")
