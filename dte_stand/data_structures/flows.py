import uuid
from pydantic import BaseModel, validator, PositiveInt, Field

import logging
LOG = logging.getLogger(__name__)


def struuid():
    return str(uuid.uuid4())


class Flow(BaseModel):
    start: str
    end: str
    all_bandwidth: dict[str, PositiveInt]
    start_time: int
    end_time: int
    bandwidth: int = None
    flow_id: str = Field(default_factory=struuid)

    @validator('end')
    def start_differs_from_end(cls, v, values):
        if v == values['start']:
            raise ValueError('Flow start and end points are same')
        return v

    @validator('end_time')
    def start_before_end(cls, v, values):
        if v <= values['start_time']:
            raise ValueError('Flow start and end times are incorrect')
        return v

    class Config:
        exclude = {"flow_id", "bandwidth"}


class InputFlows(BaseModel):
    __root__: list[Flow]

    def __iter__(self):
        return iter(self.__root__)

    def append(self, item):
        self.__root__.append(item)

    def __getitem__(self, item):
        return self.__root__[item]


class Flows:
    def __init__(self, path_to_flows):
        self._flows: InputFlows = InputFlows.parse_file(path_to_flows)

    def get(self, current_time: int) -> list[Flow]:
        try:
            needed_flows = (f for f in self._flows if f.start_time <= current_time < f.end_time)
            result_flows = []
            for flow in needed_flows:
                latest_change = [t for t in flow.all_bandwidth if int(t) <= current_time][-1]
                flow.bandwidth = flow.all_bandwidth[latest_change]
                result_flows.append(flow)
            return result_flows
        except KeyError:
            return []
