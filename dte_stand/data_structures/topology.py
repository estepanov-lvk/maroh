import networkx
from typing import Optional
from pydantic import BaseModel
import copy


class MissingElements(BaseModel):
    missing_nodes: list[str]
    missing_links: list[tuple[str, str, int]]


class TopologyChanges(BaseModel):
    __root__: dict[str, MissingElements]

    def __iter__(self):
        return iter(self.__root__)

    def __getitem__(self, item):
        return self.__root__[item]

    def keys(self):
        return self.__root__.keys()


class Topology:
    def __init__(self, path_to_graph: str, path_to_graph_changes: str):
        with open(path_to_graph, 'rb') as file_graph:
            self._initial_topology: networkx.MultiDiGraph = networkx.readwrite.read_gml(file_graph)

        self._topology_changes = TopologyChanges.parse_file(path_to_graph_changes)
        self._changed_at: list[str] = list(self._topology_changes.keys())
        self._changed_at.sort()
        # remember the last time get was called. Used for determining the change to topology to apply
        self._previous_time = -1

    def get(self, current_time: int) -> tuple[networkx.MultiDiGraph, Optional[int]]:
        current_topology: networkx.MultiDiGraph = copy.deepcopy(self._initial_topology)

        # get time of latest change. It is the first point of change between previous time and current time
        try:
            latest_change = [t for t in self._changed_at
                             if self._previous_time < int(t) <= current_time][0]
        except IndexError:
            # no changes in topology between previous and current
            # so take the last change before previous time
            try:
                latest_change = [t for t in self._changed_at if int(t) <= self._previous_time][-1]
            except IndexError:
                # no changes at all
                return current_topology, None
        self._previous_time = int(latest_change)

        # apply change
        try:
            current_changes = self._topology_changes[str(latest_change)]
        except KeyError:
            # no changes in topology at given time slot
            return current_topology, None

        for node_id in current_changes.missing_nodes:
            current_topology.remove_node(node_id)
        for node1_id, node2_id, index in current_changes.missing_links:
            current_topology.remove_edge(node1_id, node2_id, index)
        return current_topology, int(latest_change)
