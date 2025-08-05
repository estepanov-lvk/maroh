import os
import json
import copy
from numpy import ndarray, float32
from tensorflow import Tensor
from collections import defaultdict
from dte_stand.config import Config


class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ndarray):
            return obj.tolist()
        if isinstance(obj, Tensor):
            return obj.numpy()
        if isinstance(obj, float32):
            return obj.item()
        return json.JSONEncoder.default(self, obj)


class HistoryTracker:
    _result_folder = './'

    @classmethod
    def set_result_folder(cls, folder):
        cls._result_folder = folder

    def __init__(self):
        config = Config.config()
        self._plot_period = config.plot_period
        self._iteration = 0
        self._history: dict[str, list] = defaultdict(lambda: [])
        self._plot_number = 0

    def add_value(self, tracked_name, value):
        self._history[tracked_name].append(copy.deepcopy(value))

    def _save_to_file(self):
        self._plot_number += 1
        for name, values in self._history.items():
            values_json = json.dumps(values, cls=TensorEncoder)
            with open(os.path.join(
                        HistoryTracker._result_folder, f'{name}_'
                                                       f'{(self._plot_number - 1) * self._plot_period}-'
                                                       f'{self._plot_number * self._plot_period - 1}.json'), 'w') as f:
                f.write(values_json)

    def end_iteration(self):
        self._iteration += 1
        if self._iteration % self._plot_period == 0:
            self._save_to_file()
            self._history.clear()

    def reset(self):
        self._iteration = 0
        self._plot_number = 0
        self._history.clear()