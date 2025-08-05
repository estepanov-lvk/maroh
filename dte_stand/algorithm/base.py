import abc
import networkx
from typing import Callable
from dte_stand.data_structures import HashWeights, Flow
from dte_stand.hash_function.base import BaseHashFunction


class BaseAlgorithm(metaclass=abc.ABCMeta):
    def __init__(self, hash_function: BaseHashFunction, phi_func: Callable, experiment_dir: str, model_dir: str):
        self._hash_function = hash_function
        self._phi = phi_func
        self._experiment_dir = experiment_dir
        self._model_dir = model_dir

    @abc.abstractmethod
    def step(self, topology: networkx.MultiDiGraph, flows: list[Flow],
             iteration_num: int = 0, save_model: bool = False) -> HashWeights:
        """
        Main function for algorithm

        :param topology: current topology. May be used freely
        :param flows: list of current flows. Flow parameters cannot be used by algorithm.
            Flows are passed only for cases when hash function needs to use them to recalculate current bandwidth
        :param iteration_num: number of current iteration in the experiment
        :param save_model: fpr ML algorithms, whether to save the current model to disk for later use
        :return: final hash weights
        """
        ...
