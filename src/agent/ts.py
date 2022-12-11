import collections
import numpy

from src.agent import agent
from src.prob import random_variable
from src.utils.debug import *


class AssignWithThompsonSampling_slidingWin(agent.SchingAgent):
    def __init__(self, node_id_list: list[str], win_len: int):
        super().__init__(node_id_list=node_id_list)

        self.win_len = win_len

        self.node_id_cost_queue = collections.deque(maxlen=win_len)

    def __repr__(self):
        return (
            "AssignWithThompsonSampling_slidingWin( \n"
            f"{super().__repr__()} \n"
            f"\t win_len= {self.win_len} \n"
            ")"
        )

    def record_cost(self, node_id: str, cost: float):
        self.node_id_cost_queue.append((node_id, cost))
        log(DEBUG, "recorded", node_id=node_id, cost=cost)

    def node_to_assign(self):
        # Construct `node_id_to_costs_map`
        node_id_to_costs_map = collections.defaultdict(list)
        for (node_id, cost) in self.node_id_cost_queue:
            node_id_to_costs_map[node_id].append(cost)

        for node_id in self.node_id_list:
            if node_id not in node_id_to_costs_map:
                node_id_to_costs_map[node_id].append(0)

        log(DEBUG, "", node_id_to_costs_map=node_id_to_costs_map)

        # Choose the node with min cost sample
        node_id_w_min_sample, min_sample = None, float("Inf")
        for node_id, cost_queue in node_id_to_costs_map.items():
            mean = numpy.mean(cost_queue) if len(cost_queue) else 0
            stdev = numpy.std(cost_queue) if len(cost_queue) else 1
            check(stdev >= 0, "Stdev cannot be negative")
            if stdev == 0:
                stdev = 1

            s = random_variable.TruncatedNormal(mu=mean, sigma=stdev).sample()
            if s < min_sample:
                min_sample = s
                node_id_w_min_sample = node_id
                # log(DEBUG, "s < min_sample", s=s, min_sample=min_sample, node_id_w_min_sample=node_id_w_min_sample)

        return node_id_w_min_sample


class AssignWithThompsonSampling_slidingWinForEachNode(agent.SchingAgent):
    def __init__(self, node_id_list: list[str], win_len: int):
        super().__init__(node_id_list=node_id_list)

        self.win_len = win_len

        self.node_id_to_cost_queue_map = {node_id: collections.deque(maxlen=win_len) for node_id in node_id_list}

    def __repr__(self):
        return (
            "AssignWithThompsonSampling_slidingWinForEachNode( \n"
            f"{super().__repr__()} \n"
            f"\t win_len= {self.win_len} \n"
            ")"
        )

    def record_cost(self, node_id: str, cost: float):
        self.node_id_to_cost_queue_map[node_id].append(cost)
        log(DEBUG, "recorded", node_id=node_id, cost=cost)

    def node_to_assign(self):
        log(DEBUG, "", node_id_to_cost_queue_map=self.node_id_to_cost_queue_map)

        # Choose the node with min cost sample
        node_id_w_min_sample, min_sample = None, float("Inf")
        for node_id, cost_queue in self.node_id_to_cost_queue_map.items():
            mean = numpy.mean(cost_queue) if len(cost_queue) else 0
            stdev = numpy.std(cost_queue) if len(cost_queue) else 1
            check(stdev >= 0, "Stdev cannot be negative")
            if stdev == 0:
                stdev = 1

            s = random_variable.TruncatedNormal(mu=mean, sigma=stdev).sample()
            if s < min_sample:
                min_sample = s
                node_id_w_min_sample = node_id
                # log(DEBUG, "s < min_sample", s=s, min_sample=min_sample, node_id_w_min_sample=node_id_w_min_sample)

        return node_id_w_min_sample
