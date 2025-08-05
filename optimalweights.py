from typing import Optional
from networkx import product
from sys import maxsize
import networkx as nx
import numpy as np
import pandas as pd
import random
import os
import sys
from datetime import datetime

from dte_stand.config import Config

from dte_stand.data_structures import HashWeights, InputData
from dte_stand.hash_function import WeightedDxHashFunction
from dte_stand.phi_calculator import PhiCalculator
from dte_stand.paths import DAGCalculator

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Genetic Algorithm Parameters')

    parser.add_argument("dir", type=str,
        help="path to directory with topology.gml, flows.json")

    parser.add_argument("--mode", type=int, choices=[1, 2], default=1,
        help="1 -- find optimal weights for links, 2 -- find optimal weights for links x destinations (default: 1)")

    parser.add_argument("--start", type=int, default=0,
        help="time_start * flowsperiod will be the starting timestamp (default: 0)")

    parser.add_argument("--end", type=int,
        help="time_end * flowsperiod will be the ending timestamp (default: time_start + 1)")

    parser.add_argument('--ntrain', type=int, default=1,
        help='Number of hash seeds for training (default: 1)')
    parser.add_argument('--ntest', type=int, default=1,
        help='Number of hash seeds for testing (default: 1)')
    parser.add_argument('--nvalid', type=int, default=50,
        help='Number of hash seeds for validation (default: 50)')

    parser.add_argument('--seed', type=int,
        help='Random seed for generating hash function seeds')
    parser.add_argument('--maxweight', type=int, default=64,
        help='Maximum weight (default: 64)')

    parser.add_argument('--flowsperiod', type=int, default=30000,
        help='Period of changes in input flows, ms. Should be equal to lsdb_period in MAROH\'s config, and value of period parameter in flow_generator.generate call in generator.py) (default: 30000)')

    parser.add_argument('--iter', type=int, default=100,
        help='Number of genetic iterations (default: 100)')
    parser.add_argument('--size', type=int, default=100,
        help='Size of the population (default: 100)')

    parser.add_argument('--crossrate', type=float, default=0.7,
        help='Crossover rate (default: 0.7)')
    parser.add_argument('--mutrate', type=float, default=0.7,
        help='Mutation rate (default: 0.7)')
    parser.add_argument('--swaps', type=int, default=5,
        help='Number of crossover swaps (default: 5)')
    parser.add_argument('--shifts', type=int, default=4,
        help='Number of mutation shifts (default: 4)')
    parser.add_argument('--reassigns', type=int, default=4,
        help='Number of mutation reassigns (how many weights to reassign in chosen row/column per mutation) (default: 4)')

    args = parser.parse_args()
    return args

def population_generation(cost):
    """
    Random population generation
    """
    population = []
    for i in range(POPULATION_SIZE):
        population.append(np.zeros_like(cost))
    for i in range(cost.shape[0]):
        for j in range(cost.shape[1]):
            if cost[i,j] != 0:
                for k in range(POPULATION_SIZE):
                    population[k][i][j] = np.random.randint(1, MAX_WEIGHT+1)
    # for i in range(cost.shape[0]):
    #     for j in range(cost.shape[1]):
    #             for k in range(POPULATION_SIZE):
    #                 population[k][i][j] = population[k][j][i]
    return population

def selection_direct(phi_list):
    """
    Selection using roulette wheel with probability ~ 1 / fitness
    """
    choice_weights = np.array(list(map(lambda x: 1/x, phi_list)))
    choice_weights /= np.sum(choice_weights)
    new_population_idx = np.random.choice(np.arange(len(population), dtype=int), POPULATION_SIZE, p=choice_weights)
    new_population = [population[i] for i in new_population_idx]
    population.clear()
    population.extend(new_population)

def selection(population, phi_list):
    """
    Selection using linear ranking
    """
    n = len(phi_list)
    ranks = np.argsort(np.argsort(phi_list))
    sp = 1.9 # sp in [1,2]. the bigger is sp, the more different are weights
    linear_ranking = lambda i: (sp - (2 * sp - 2) * i / (n-1)) / n
    p = list(map(linear_ranking, ranks))
    new_population_idx = np.random.choice(np.arange(len(population), dtype=int), POPULATION_SIZE, p=p)
    new_population = [population[i] for i in new_population_idx]
    population.clear()
    population.extend(new_population)

def crossover_no_nexthops(population):
    """
    Crossover
    """
    num = len(population) - (len(population) % 2)

    for i in range(0, num, 2):
        if np.random.rand() <= CROSSOVER_RATE:
            child1: np.ndarray = population[i].copy()
            child2: np.ndarray = population[i+1].copy()
            comparison: np.ndarray = (child1 == child2)
            if comparison.all():
                continue
            for it in range(CROSSOVER_SWAPS):
                crossover_row = np.random.randint(0, len(child1))
                if np.random.rand() <= 0.5:
                    # doing for row
                    t1 = child1[crossover_row].copy()
                    t2 = child2[crossover_row].copy()
                    child1[crossover_row] = t2
                    child2[crossover_row] = t1
                else:
                    # doing for column
                    t3 = child1[:, crossover_row].copy()
                    t4 = child2[:, crossover_row].copy()
                    child1[:, crossover_row] = t4
                    child2[:, crossover_row] = t3
            population.append(child1)
            population.append(child2)

def crossover_with_nexthops(population):
    """
    Crossover
    """
    crossover_no_nexthops(population)

def crossover(population, problem_mode):
    if problem_mode == 1:
        crossover_no_nexthops(population)
    elif problem_mode == 2:
        crossover_with_nexthops(population)
    else:
        raise Exception("invalid problem mode")

def mutation(population, problem_mode):
    if problem_mode == 1:
        mutation_no_nexthops(population)
    elif problem_mode == 2:
        mutation_with_nexthops(population)
    else:
        raise Exception("invalid problem mode")

def mutation_no_nexthops(population):
    """
    Shift mutation
    """
    for i in range(0, len(population)):
        if np.random.rand() <= MUTATION_RATE:
            new_s = population[i].copy()
            n = np.random.randint(0, new_s.shape[0])
            for it in range(MUTATION_SHIFTS):
                # cyclic shift the row's (or column's) non-zero elements by one element,
                # and reassign its random elements to new random values;
                rowwise = (np.random.rand() <= 0.5)
                if not rowwise:
                    new_s = new_s.T
                idx_nonzero = np.where(new_s[n] != 0)
                if np.random.rand() <= 0.5:
                    # shift
                    new_s[n][idx_nonzero] = np.roll(new_s[n][idx_nonzero], -1)
                for it2 in range(MUTATION_REASSIGNS):
                    m = np.random.randint(0, len(idx_nonzero))
                    new_s[n][idx_nonzero[m]] = np.random.randint(1, MAX_WEIGHT+1)
                if not rowwise:
                    new_s = new_s.T
            population.append(new_s)

def mutation_with_nexthops(population):
    """
    Shift mutation
    """
    for i in range(0, len(population)):
        if np.random.rand() <= MUTATION_RATE:
            new_s = population[i].copy()
            n = np.random.randint(0, new_s.shape[0])
            for it in range(MUTATION_SHIFTS):
                # cyclic shift the row's (or column's) non-zero elements by one element,
                # and reassign its random elements to new random values;
                rowwise = (np.random.rand() <= 0.5)
                if not rowwise:
                    new_s = new_s.T
                idx_nonzero = np.where(new_s[n] != 0)
                if np.random.rand() <= 0.5:
                    # shift
                    new_s[n][idx_nonzero] = np.roll(new_s[n][idx_nonzero], -1)
                for it2 in range(MUTATION_REASSIGNS * (n if rowwise else 1)):
                    m = np.random.randint(0, len(idx_nonzero))
                    new_s[n][idx_nonzero[m]] = np.random.randint(1, MAX_WEIGHT+1)
                if not rowwise:
                    new_s = new_s.T
            population.append(new_s)

def convert_to_hash_weights(weight_matr, problem_mode):
    global n_nodes, node_labels
    hash_weights = HashWeights()
    n = n_nodes
    if problem_mode == 1:
        for i in range(n):
            for j in range(n):
                if weight_matr[i][j] != 0:
                    for k in range(n):
                        hash_weights.put(node_labels[i], node_labels[k], node_labels[j], 0, weight_matr[i][j])
    elif problem_mode == 2:
        for i in range(n):
            for j in range(n):
                if weight_matr[i][j * n] != 0:
                    for k in range(n):
                        assert(weight_matr[i][j * n + k] != 0)
                        hash_weights.put(node_labels[i], node_labels[k], node_labels[j], 0, weight_matr[i][j * n + k])
    return hash_weights

def calc_phi(hash_weights, hash_seed=123):
    global path_calculator, phi_func, topo_orig, flows, hash_functions
    topo_changed = topo_orig.copy()
    # path_calculator_2 = DAGCalculator()
    hash_function = WeightedDxHashFunction(path_calculator, seed=hash_seed)  # reset
    # hash_function = hash_functions[hash_seed]
    # hash_function.path_calculator.prepare_iteration(topo_changed)
    # path_calculator.prepare_iteration(topo_changed)
    hash_function.run(topo_changed, flows, hash_weights)
    CBandwidth = pd.DataFrame(0, index=list(topo_changed.nodes),
                              columns=list(topo_changed.nodes))
    for edge in topo_changed.edges(data=True):
        CBandwidth.loc[edge[0], edge[1]] += edge[-1]['current_bandwidth']
    phi = phi_func(topo_changed)
    return phi, CBandwidth

def calc_phi_multi(weights, hash_seeds, problem_mode):
    hash_weights = convert_to_hash_weights(weights, problem_mode)
    phi_tries = []
    cbw_tries = []
    CBandwidth = None
    for hash_seed in hash_seeds:
        phi, CBandwidth = calc_phi(hash_weights, hash_seed=hash_seed)
        cbw_tries.append(CBandwidth)
        phi_tries.append(phi)
    return phi_tries, cbw_tries

def phi_stats(phi_list):
    phi_mean = np.mean(phi_list)
    phi_std = np.std(phi_list)
    return phi_mean, phi_std

def cbw_stats(cbw_list):
    CBandwidth_example = cbw_list[0]
    cbw_list_np = np.array([CBandwidth.to_numpy() for CBandwidth in cbw_list])

    CBandwidth_mean = pd.DataFrame(np.mean(cbw_list_np, axis=0), index=CBandwidth_example.index,
                                   columns=CBandwidth_example.columns)
    CBandwidth_std = pd.DataFrame(np.std(cbw_list_np, axis=0), index=CBandwidth_example.index,
                                  columns=CBandwidth_example.columns)
    return CBandwidth_mean, CBandwidth_std

def fitness(population, hash_seeds, problem_mode):
    """
    Phi calculation (using existing modules)
    """
    phi_lists = []
    phi_mean_list = []
    phi_std_list = []
    cbw_lists = [] # current bandwidths dataframe for each population (mean over hash_seed tries)
    cbw_mean_list = []
    cbw_std_list = []
    for weights in population:
        phi_list, cbw_list = calc_phi_multi(weights, hash_seeds, problem_mode)
        phi_mean, phi_std = phi_stats(phi_list)
        CBandwidth_mean, CBandwidth_std = cbw_stats(cbw_list)
        phi_lists.append(phi_list)
        cbw_lists.append(cbw_list)
        phi_mean_list.append(phi_mean)
        phi_std_list.append(phi_std)
        cbw_mean_list.append(CBandwidth_mean)
        cbw_std_list.append(CBandwidth_std)
    return phi_lists, phi_mean_list, phi_std_list, cbw_lists, cbw_mean_list, cbw_std_list

if __name__ == "__main__":
    # Genetic algorithm (GA) will use average of Ф value over the set of
    # training hash functions as a METRIC of a candidate.
    # It will use this metric to rank candidates in a population and select best candidate.
    # After each iteration Ф will be also calculated (but not used)
    # on the set of testing hash functions for the currently best candidate.
    # After the last iteration, Ф values will be calculated (but not used) on the set
    # of validation hash functions for the overall best candidate.
    # Thus, Ф is calculated:
    # - for the WHOLE population on EVERY iteration of GA on every TRAINING hash function;
    # - once per GA iteration on every TESTING hash function;
    # - once after the last GA iteration on every VALIDATION hash function.
    # Choose number of training, testing and validation functions considering how long
    # it will take to calculate Ф that many times.

    # Output data will be written in root directory (using current timestamp to prevent overwriting).

    args = parse_args()

    GENETIC_NUM = args.iter
    POPULATION_SIZE = args.size
    CROSSOVER_RATE = args.crossrate
    MUTATION_RATE = args.mutrate
    NUM_HASH_SEEDS_TRAIN = args.ntrain
    NUM_HASH_SEEDS_TEST = args.ntest
    NUM_HASH_SEEDS_VALID = args.nvalid
    RANDOM_SEED = args.seed if args.seed is not None else random.randint(0, 1000)
    MAX_WEIGHT = args.maxweight
    CROSSOVER_SWAPS = args.swaps
    MUTATION_SHIFTS = args.shifts
    MUTATION_REASSIGNS = args.reassigns

    # Reading topology
    experiment_folder = args.dir
    if experiment_folder.endswith(os.path.sep):
        experiment_folder = experiment_folder[:-len(os.path.sep)]
    experiment_path_rel = os.path.split(experiment_folder)[-1]
    problem_mode = args.mode
    tim_start = args.start
    tim_end = args.end if args.end is not None else tim_start + 1
    flowsperiod = args.flowsperiod

    Config.load_config(experiment_folder)
    input_data = InputData(experiment_folder)
    topo_orig = input_data.topology.get(0)[0]

    print("Parameters:")
    print(f"GENETIC_NUM_ITERATIONS: {GENETIC_NUM}")
    print(f"POPULATION_SIZE: {POPULATION_SIZE}")
    print(f"MAX_WEIGHT: {MAX_WEIGHT}")

    print(f"NUM_HASH_SEEDS_TRAIN: {NUM_HASH_SEEDS_TRAIN}")
    print(f"NUM_HASH_SEEDS_TEST: {NUM_HASH_SEEDS_TEST}")
    print(f"NUM_HASH_SEEDS_VALID: {NUM_HASH_SEEDS_VALID}")
    print(f"RANDOM_SEED: {RANDOM_SEED}")

    print(f"EXPERIMENT_FOLDER: {experiment_folder}")
    print(f"PROBLEM_MODE: {problem_mode}")
    print(f"TIME_START: {tim_start}")
    print(f"TIME_END: {tim_end}")
    print(f"FLOWS_PERIOD: {flowsperiod}")

    print(f"CROSSOVER_RATE: {CROSSOVER_RATE}")
    print(f"MUTATION_RATE: {MUTATION_RATE}")
    print(f"CROSSOVER_SWAPS: {CROSSOVER_SWAPS}")
    print(f"MUTATION_SHIFTS: {MUTATION_SHIFTS}")
    print(f"MUTATION_REASSIGNS: {MUTATION_REASSIGNS}")

    # topo_read = nx.read_gml(os.path.join(experiment_folder, "topology.gml"), destringizer=int)
    # topo_convert = nx.convert_node_labels_to_integers(topo_read)
    # topo_dict = nx.to_dict_of_dicts(topo_convert)
    # number_of_edges = topo_convert.number_of_edges()
    # nodes = list(topo_convert.nodes)
    # n_nodes = len(nodes)

    n_nodes = len(list(topo_orig.nodes))
    node_idx = dict(zip(
        list(topo_orig.nodes), range(n_nodes)
    ))
    node_labels = dict(zip(
        range(n_nodes), list(topo_orig.nodes)
    ))

    # with open(os.path.join(experiment_folder, "flows.json"), 'r') as f:
    #     flows = json.load(f)

    # flows_multiply_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    plot_data = []
    # population = list()
    # optimal_sol = None
    min_phi = maxsize
    # cost = np.array(cost)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # for i in range(flows_multiply - 1):
    #     flows_add = InputData(experiment_folder).flows.get(0)
    #     flows.extend(flows_add)
    # assert(len(input_data.flows.get(0)) * flows_multiply == len(flows))

    # hash_weights = convert_to_hash_weights(cost, problem_mode)
    phi_func = PhiCalculator.calculate_phi
    path_calculator = DAGCalculator()

    # first training seed will be same as in usual MAROH run
    hash_seeds_traintest = [None] + [RANDOM_SEED*(i+1) for i in range(NUM_HASH_SEEDS_TRAIN + NUM_HASH_SEEDS_TEST + NUM_HASH_SEEDS_VALID - 1)]
    hash_seeds_train = hash_seeds_traintest[:NUM_HASH_SEEDS_TRAIN]
    hash_seeds_test = hash_seeds_traintest[NUM_HASH_SEEDS_TRAIN:NUM_HASH_SEEDS_TRAIN+NUM_HASH_SEEDS_TEST]
    hash_seeds_valid = hash_seeds_traintest[-NUM_HASH_SEEDS_VALID:]

    # hash_functions = dict()s
    # for hash_seed in hash_seeds_traintest:
    #     path_calculator_2 = DAGCalculator()
    #     hash_functions[hash_seed] = WeightedDxHashFunction(path_calculator=path_calculator_2)

    if problem_mode == 1:
        cost = np.zeros((n_nodes, n_nodes), dtype=int)
        for edge in topo_orig.edges(data=True):
            if edge[-1]['bandwidth'] != 0:
                i = node_idx[edge[0]]
                j = node_idx[edge[1]]
                cost[i, j] = 1
        # for i in range(n_nodes):
        #     for k in topo_dict[nodes[i]].keys():
        #         cost[i, k] = 1
    else:
        cost = np.zeros((n_nodes, n_nodes*n_nodes), dtype=int)
        for edge in topo_orig.edges(data=True):
            if edge[-1]['bandwidth'] != 0:
                i = node_idx[edge[0]]
                j = node_idx[edge[1]]
                for k in range(n_nodes):
                    cost[i, j * n_nodes + k] = 1
        # for i in range(n_nodes):
        #     for k in topo_dict[nodes[i]].keys():
        #         for j in range(n_nodes):
        #             cost[i, k * n_nodes + j] = 1
    print(cost)

    path_calculator.prepare_iteration(topo_orig)

    population = population_generation(cost)

    Bandwidth = pd.DataFrame(0, index=list(topo_orig.nodes),
                              columns=list(topo_orig.nodes))

    for edge in topo_orig.edges(data=True):
        assert(Bandwidth.loc[edge[0], edge[1]] == 0) # otherwise it's a multigraph, not supported yet
        Bandwidth.loc[edge[0], edge[1]] += edge[-1]['bandwidth']

    for tim in range(tim_start * flowsperiod, tim_end * flowsperiod, flowsperiod):
        print("env time:", tim)
        t1_0 = datetime.now()
        flows = input_data.flows.get(tim)
        print("number of flows:", len(flows))
        optimal_value = 1e10
        optimal_sol = population[0]
        optimal_value_std = 0
        optimal_value_test_mean = 0
        optimal_value_test_std = 0
        optimal_sol_cbw_str = (Bandwidth/Bandwidth).astype(str)
        phi_mean_list = []
        n_phi_calcs = 0
        flowset_time_elapsed = 0.0

        dat_str = datetime.now().isoformat().replace(':', '-')
        filename_prefix = f"ow_exp_problem={problem_mode}-{experiment_path_rel}-{tim}"

        """
        GA algorithm steps
        """
        try:
            for it in range(GENETIC_NUM+1):
                t1 = datetime.now()
                if it > 0:
                    selection(population, phi_mean_list)
                    crossover(population, problem_mode)
                    mutation(population, problem_mode)
                phi_lists, phi_mean_list, phi_std_list, \
                    cbw_lists, cbw_mean_list, cbw_std_list = fitness(population, hash_seeds_train, problem_mode)
                n_phi_calcs += len(population) * NUM_HASH_SEEDS_TRAIN
                t2 = datetime.now()
                flowset_time_elapsed += (t2-t1).total_seconds()
                min_sol_idx = np.argmin(phi_mean_list)
                min_phi = phi_mean_list[min_sol_idx]
                min_phi_list = phi_lists[min_sol_idx]
                min_sol = population[min_sol_idx]
                min_phi_std = phi_std_list[min_sol_idx]
                if hash_seeds_test:
                    min_phi_test_list, _ = calc_phi_multi(min_sol, hash_seeds_test, problem_mode)
                else:
                    min_phi_test_list = [0]
                min_phi_test_mean, min_phi_test_std = phi_stats(min_phi_test_list)
                min_cbw_mean = cbw_mean_list[min_sol_idx]
                min_cbw_std = cbw_std_list[min_sol_idx]
                min_cbw_str = pd.DataFrame('', index=min_cbw_mean.index,
                                                columns=min_cbw_mean.columns)
                for i in min_cbw_str.index:
                    for j in min_cbw_mean.columns:
                        x1 = (min_cbw_mean.loc[i, j] / Bandwidth.loc[i, j]) if Bandwidth.loc[i, j] > 0 else np.nan
                        x2 = (min_cbw_std.loc[i, j] / Bandwidth.loc[i, j]) if Bandwidth.loc[i, j] > 0 else np.nan
                        min_cbw_str[i][j] = f"{x1:6.3f}±{x2:5.3f}"
                if min_phi < optimal_value:
                    optimal_value = min_phi
                    optimal_sol = min_sol
                    optimal_value_std = min_phi_std
                    optimal_value_test_mean = min_phi_test_mean
                    optimal_value_test_std = min_phi_test_std
                    optimal_sol_cbw_str = min_cbw_str
                if it == 0:
                    np.save(f'{filename_prefix}_{dat_str}_iter0_population.npy', np.array(phi_mean_list))
                phi_list_str = " ".join([f"{phi_mean:6.4f}±{phi_std:6.4f}, "
                                         for phi_mean, phi_std in zip(phi_mean_list, phi_std_list)]).ljust(150 * 11)
                print(f"{it:2}:\n best weight matrix in population:\n", min_sol)
                print(f"{it:2}: best phi = {min_phi:7.5f}±{min_phi_std:6.4f} (for other hash functions: {min_phi_test_mean:6.4f}±{min_phi_test_std:6.4f})")
                phi_min_train_list_str = ', '.join([f"{phi:.8f}" for phi in min_phi_list])
                phi_min_test_list_str = ', '.join([f"{phi:.8f}" for phi in min_phi_test_list])
                print(f"{it:2}: best phi (all hash functions): [{phi_min_train_list_str}] (for other hash functions: [{phi_min_test_list_str}])")
                print("links load:\n", min_cbw_str)
                print("population len =", len(population))
                print("the whole population:", phi_list_str)
                print()
                print()
                for phi in min_phi_list:
                    plot_data.append((experiment_path_rel, len(flows), problem_mode, tim, it, phi, "train"))
                for phi in min_phi_test_list:
                    plot_data.append((experiment_path_rel, len(flows), problem_mode, tim, it, phi, "test"))
        except KeyboardInterrupt:
            print("================ KeyboardInterrupt")

        print("Phi-function value:", optimal_value)
        print("Optimal solution:\n", optimal_sol)
        print(f"Phi-function value: {optimal_value:7.5f}±{optimal_value_std:6.4f}"
                  + f" (for other hash functions: {optimal_value_test_mean:6.4f}±{optimal_value_test_std:6.4f})")
        print("links load:\n", optimal_sol_cbw_str)
        min_phi_list, _ = calc_phi_multi(optimal_sol, hash_seeds_train, problem_mode)
        min_phi_test_list, _ = calc_phi_multi(optimal_sol, hash_seeds_valid, problem_mode)
        # phi_min_train_list_str = ', '.join([f"{phi:.8f}" for phi in min_phi_list])
        # phi_min_test_list_str = ', '.join([f"{phi:.8f}" for phi in min_phi_test_list])
        # print(f"mul=: best phi = {min_phi:7.5f}±{min_phi_std:6.4f} (for other hash functions: {min_phi_test_mean:6.4f}±{min_phi_test_std:6.4f})")
        for phi in min_phi_list:
            plot_data.append((experiment_path_rel, len(flows), problem_mode, tim, -1, phi, "train"))
        for phi in min_phi_test_list:
            plot_data.append((experiment_path_rel, len(flows), problem_mode, tim, -1, phi, "valid"))
        columns = ["experiment", "n_flows", "problem_mode", "env_time", "iteration", "phi_min", "hash_funcs_set"]
        df = pd.DataFrame(plot_data, columns=columns)
        df.to_csv(f"{filename_prefix}_{dat_str}.csv", index=False)
        np.save(f'{filename_prefix}_{dat_str}_solution.npy', optimal_sol)
        t2_0 = datetime.now()
        print(f"time elapsed (total): {(t2_0-t1_0).total_seconds():.6f} s", )
        print(f"time elapsed (clean): {flowset_time_elapsed:.6f} s")
        print(f"({flowset_time_elapsed / (GENETIC_NUM+1):.6f} s / iter, on {GENETIC_NUM+1} iterations)")
        print(f"({flowset_time_elapsed / n_phi_calcs:.6f} s / weight matrix, on {n_phi_calcs} matrices)")
        avg_population_size = float(n_phi_calcs) / (GENETIC_NUM+1) / NUM_HASH_SEEDS_TRAIN
        print(f"population size parameter = {POPULATION_SIZE}, average actual size = {avg_population_size:.2f}")
