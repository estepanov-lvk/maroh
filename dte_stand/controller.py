import copy
import uuid as uuid_module

import networkx
import os
import random
from dte_stand.data_structures import HashWeights, Flow, InputData
from dte_stand.hash_function.base import BaseHashFunction
from dte_stand.algorithm.base import BaseAlgorithm
from dte_stand.paths.base import BasePathCalculator
from dte_stand.phi_calculator import PhiCalculator
from dte_stand.history import HistoryTracker
from dte_stand.config import Config
from typing import Optional, Callable

import logging
LOG = logging.getLogger(__name__)


class ExperimentController:
    def __init__(self, path_to_inputs: str, lsdb_period: int, num_iterations: int,
                 hash_function: BaseHashFunction, algorithm: BaseAlgorithm, path_calculator: BasePathCalculator,
                 phi_func: Callable, experiment_dir: str):
        self.input_data = InputData(path_to_inputs)
        self.period = lsdb_period
        self.num_iterations = num_iterations
        self.hash_function = hash_function
        self.algorithm = algorithm
        self.path_calculator = path_calculator
        self.phi = phi_func
        self.experiment_dir = experiment_dir
        # self.removed_start1 = ''
        # self.removed_end1 = ''
        # self.removed_start2 = ''
        # self.removed_end2 = ''

    def _init_hash_weights(self, topology: networkx.MultiDiGraph) -> HashWeights:
        hw = HashWeights()
        nodes = list(topology.nodes())
        for start, end in topology.edges():
            for dst_node in nodes:
                hw.put(start, dst_node, end, 0, 1)
        return hw

    def _get_current_topology_and_time(self, current_time: int) -> tuple[networkx.MultiDiGraph, int]:
        # get topology and time when topology last changed
        current_topology, change_time = self.input_data.topology.get(current_time + self.period)

        # if topology changed between (current_time, current_time+period),
        # then current time is actually the time when it changed,
        # because our algorithm was run because of the change, not because a period has passed
        if (change_time is not None) and (current_time < change_time < current_time + self.period):
            return current_topology, change_time

        # otherwise, no changes to topology in the experiment yet, or the change was too long ago
        return current_topology, current_time + self.period

    def _calculate_current_bandwidth(self, topology: networkx.MultiDiGraph, flows: list[Flow],
                                     hash_weights: HashWeights) -> None:
        if hash_weights is None:
            # first iteration
            return
        self.hash_function.run(topology, list(flows), hash_weights, use_flow_memory=False)

    def _end_iteration(self):
        PhiCalculator.end_iteration_and_plot_graph()
        PhiCalculator.plot_full(all_iterations=False)
        self.hash_function.end_iteration()

    def remove_two_random_links(self, topology: networkx.MultiDiGraph):
        edges1 = list(topology.edges())
        edges2 = list(edges1)
        random.shuffle(edges1)
        random.shuffle(edges2)
        self.removed_start1, self.removed_end1 = edges1[0]
        what_removed = {self.removed_end1, self.removed_start1}
        for edge in edges2:
            self.removed_start2, self.removed_end2 = edge
            if (self.removed_start2 in what_removed) or (self.removed_end2 in what_removed):
                continue
            break
        topology.remove_edge(self.removed_start1, self.removed_end1)
        topology.remove_edge(self.removed_end1, self.removed_start1)
        topology.remove_edge(self.removed_start2, self.removed_end2)
        topology.remove_edge(self.removed_end2, self.removed_start2)

    def run(self) -> float:
        hash_weights: Optional[HashWeights] = None
        current_time = -self.period
        for iteration in range(self.num_iterations):
            iteration_path = os.path.join(self.experiment_dir, f'iteration{iteration}')
            HistoryTracker.set_result_folder(iteration_path)
            PhiCalculator.set_plot_folder(iteration_path)

            current_topo, current_time = self._get_current_topology_and_time(current_time)
            if not hash_weights:
                hash_weights = self._init_hash_weights(current_topo)
            # self.remove_two_random_links(current_topo)
            LOG.info(f'current time: {current_time}')
            current_flows = self.input_data.flows.get(current_time)

            self.path_calculator.prepare_iteration(current_topo)
            self._calculate_current_bandwidth(current_topo, current_flows, hash_weights)
            phi = self.phi(current_topo)
            LOG.info(f'Iteration: {iteration}, phi: {phi}')

            hash_weights = self.algorithm.step(current_topo, current_flows, iteration_num=iteration, save_model=True)

            self._end_iteration()
        self._calculate_current_bandwidth(current_topo, current_flows, hash_weights)
        phi = self.phi(current_topo)
        LOG.info(f'phi after experiment: {phi}')
        return phi
        # return phi, self.removed_start1, self.removed_end1, self.removed_start2, self.removed_end2


class RandomExperimentController(ExperimentController):
    def __init__(self, path_to_inputs: str, lsdb_period: int, num_iterations: int,
                 hash_function: BaseHashFunction, algorithm: BaseAlgorithm, path_calculator: BasePathCalculator,
                 phi_func: Callable, experiment_dir: str):
        super().__init__(path_to_inputs, lsdb_period, num_iterations, hash_function,
                         algorithm, path_calculator, phi_func, experiment_dir)
        config = Config.config()
        self._plot_period = config.plot_period
        self._time_points = []
        random.seed() # TODO: should not change global seed, but maybe this was the idea

    def _generate_time_points(self):
        rng = random.Random()
        self._time_points = list(range(0, 1 * 30000, self.period))
        rng.shuffle(self._time_points)

    def _update_flow_ids(self, flows: list[Flow]):
        for flow in flows:
            flow.flow_id = str(uuid_module.uuid4())

    def run(self):
        current_topo, _ = self.input_data.topology.get(0)
        hash_weights = self._init_hash_weights(current_topo)
        self.path_calculator.prepare_iteration(current_topo)
        for iteration in range(self.num_iterations):
            hash_weights = self._init_hash_weights(current_topo)
            try:
                current_time = self._time_points.pop()
            except IndexError:
                self._generate_time_points()
                current_time = self._time_points.pop()
            LOG.info(f'current time: {current_time}')
            current_flows = self.input_data.flows.get(current_time)
            # self._update_flow_ids(current_flows)

            self._calculate_current_bandwidth(current_topo, current_flows, hash_weights)
            phi = self.phi(current_topo)
            LOG.info(f'Iteration: {iteration}, phi: {phi}')
            save_model = True if iteration == self.num_iterations - 1 else False

            hash_weights = self.algorithm.step(current_topo, current_flows,
                                               iteration_num=iteration, save_model=save_model)

            self.hash_function.end_iteration()
            PhiCalculator.end_episode()
            if iteration > 0 and ((iteration + 1) % self._plot_period == 0):
                PhiCalculator.plot_result()
        self._calculate_current_bandwidth(current_topo, current_flows, hash_weights)
        phi = self.phi(current_topo)
        LOG.info(f'phi after experiment: {phi}')
        PhiCalculator.plot_full(all_iterations=False)
        return phi
