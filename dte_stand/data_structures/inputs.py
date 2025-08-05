import os
from dte_stand.data_structures.topology import Topology
from dte_stand.data_structures.flows import Flows


class InputData:
    def __init__(self, path_to_folder):
        self._flows = Flows(os.path.join(path_to_folder, 'flows.json'))
        self._topology = Topology(os.path.join(path_to_folder, 'topology.gml'),
                                  os.path.join(path_to_folder, 'topology_changes.json'))

    @property
    def topology(self):
        return self._topology

    @property
    def flows(self):
        return self._flows
