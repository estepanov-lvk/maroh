import networkx
import os
from typing import Optional, Callable
from dte_stand.hash_function.base import BaseHashFunction
from dte_stand.algorithm.base import BaseAlgorithm
from dte_stand.data_structures import HashWeights, Flow
from dte_stand.algorithm.mate.lib.run_experiment import Runner

import logging
LOG = logging.getLogger(__name__)


class MateAlgorithm(BaseAlgorithm):
    def __init__(self, hash_function: BaseHashFunction, phi_func: Callable, experiment_dir: str, model_dir: str, multi_actions: bool = False):
        super().__init__(hash_function, phi_func, experiment_dir, model_dir)
        self._runner: Optional[Runner] = None
        self._multi_actions = multi_actions

    def step(self, topology: networkx.MultiDiGraph, flows: list[Flow],
             iteration_num: int = 0, save_model: bool = False) -> HashWeights:
        if not self._runner:
            self._runner = Runner(
                    topology, self._hash_function, self._phi,
                    checkpoint_dir=os.path.join(self._experiment_dir, f'iteration{iteration_num}', 'model'),
                    save_checkpoints=False, reload_model=bool(self._model_dir), model_dir=self._model_dir,
                    multi_actions=self._multi_actions
            )
        self._runner.update(topology, os.path.join(self._experiment_dir, f'iteration{iteration_num}', 'model'), save_model)

        return self._runner.run_experiment(topology, flows)
