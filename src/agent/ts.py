import collections
import numpy

from typing import Tuple

from src.agent import agent
from src.prob import random_variable
from src.utils.debug import *


class AssignWithThompsonSampling_slidingWin(agent.SchingAgent_wOnlineLearning):
    def __init__(self, node_id_list: list[str], win_len: int):
        super().__init__(node_id_list=node_id_list)
        self.win_len = win_len

        self.node_id_cost_queue = collections.deque(maxlen=win_len)

    def __repr__(self):
        return (
            "AssignWithThompsonSampling_slidingWin( \n"
            f"\t node_id_list= {self.node_id_list} \n"
            f"\t win_len= {self.win_len} \n"
            ")"
        )

    def record_cost(self, node_id: str, cost: float):
        self.node_id_cost_queue.append((node_id, cost))
        log(DEBUG, "recorded", node_id=node_id, cost=cost)

    def node_id_to_assign(self, time_epoch: float=None):
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


class AssignWithThompsonSampling_slidingWinForEachNode(agent.SchingAgent_wOnlineLearning):
    def __init__(self, node_id_list: list[str], win_len: int):
        super().__init__(node_id_list=node_id_list)
        self.win_len = win_len

        self.node_id_to_cost_queue_map = {node_id: collections.deque(maxlen=win_len) for node_id in node_id_list}

    def __repr__(self):
        return (
            "AssignWithThompsonSampling_slidingWinForEachNode( \n"
            f"\t node_id_list= {self.node_id_list} \n"
            f"\t win_len= {self.win_len} \n"
            ")"
        )

    def mean_stdev_cost(self, node_id: str) -> Tuple[float, float]:
        cost_queue = self.node_id_to_cost_queue_map[node_id]
        mean = numpy.mean(cost_queue) if len(cost_queue) else 0
        stdev = numpy.std(cost_queue) if len(cost_queue) else 0.01
        check(stdev >= 0, "Stdev cannot be negative")
        if stdev == 0:
            stdev = 0.01

        return mean, stdev

    def record_cost(self, node_id: str, cost: float):
        self.node_id_to_cost_queue_map[node_id].append(cost)
        log(DEBUG, "recorded", node_id=node_id, cost=cost)

    def node_id_to_assign(self, time_epoch: float=None):
        log(DEBUG, "", node_id_to_cost_queue_map=self.node_id_to_cost_queue_map)

        # Choose the node with min cost sample
        node_id_w_min_sample, min_sample = None, float("Inf")
        for node_id in self.node_id_to_cost_queue_map:
            mean, stdev = self.mean_stdev_cost(node_id)

            s = random_variable.TruncatedNormal(mu=mean, sigma=stdev).sample()
            if s < min_sample:
                min_sample = s
                node_id_w_min_sample = node_id
                # log(DEBUG, "s < min_sample", s=s, min_sample=min_sample, node_id_w_min_sample=node_id_w_min_sample)

        return node_id_w_min_sample


class AssignWithThompsonSampling_resetWinOnRareEvent(AssignWithThompsonSampling_slidingWinForEachNode):
    def __init__(self, node_id_list: list[str], win_len: int, threshold_prob_rare: float):
        super().__init__(node_id_list=node_id_list, win_len=win_len)
        self.threshold_prob_rare = threshold_prob_rare

        self.node_id_to_time_last_assigned_map = {node_id: 0 for node_id in node_id_list}

    def __repr__(self):
        return (
            "AssignWithThompsonSampling_ResetWinOnRareEvent( \n"
            f"\t node_id_list= {self.node_id_list} \n"
            f"\t win_len= {self.win_len} \n"
            f"\t threshold_prob_rare= {self.threshold_prob_rare} \n"
            ")"
        )

    def record_cost(self, node_id: str, cost: float):
        def record():
            self.node_id_to_cost_queue_map[node_id].append(cost)
            log(DEBUG, "recorded", node_id=node_id, cost=cost)

        if len(self.node_id_to_cost_queue_map[node_id]) < 5:
            record()

        else:
            mean, stdev = self.mean_stdev_cost(node_id)
            cost_rv = random_variable.TruncatedNormal(mu=mean, sigma=stdev)

            Pr_getting_larger_than_cost = cost_rv.tail_prob(cost)
            Pr_getting_smaller_than_cost = cost_rv.cdf(cost)
            Pr_cost_is_rare = 1 - min(Pr_getting_larger_than_cost, Pr_getting_smaller_than_cost)
            if Pr_cost_is_rare >= self.threshold_prob_rare:
                log(DEBUG, "Rare event detected", cost=cost, mean=mean, stdev=stdev, Pr_cost_is_rare=Pr_cost_is_rare, threshold_prob_rare=self.threshold_prob_rare)
                self.node_id_to_cost_queue_map[node_id].clear()
            else:
                record()

    def node_id_to_assign(self, time_epoch: float):
        log(DEBUG, "", node_id_to_cost_queue_map=self.node_id_to_cost_queue_map)

        # Choose the node with min cost sample
        node_id_w_min_sample, min_sample = None, float("Inf")
        for node_id in self.node_id_to_cost_queue_map:
            mean, stdev = self.mean_stdev_cost(node_id)
            if mean >= time_epoch - self.node_id_to_time_last_assigned_map[node_id]:
                log(DEBUG, "Mean >= time_epoch, resetting memory buffer", node_id=node_id)
                self.node_id_to_cost_queue_map[node_id].clear()
                node_id_w_min_sample = node_id
                break

            s = random_variable.TruncatedNormal(mu=mean, sigma=stdev).sample()
            if s < min_sample:
                min_sample = s
                node_id_w_min_sample = node_id
                # log(DEBUG, "s < min_sample", s=s, min_sample=min_sample, node_id_w_min_sample=node_id_w_min_sample)

        self.node_id_to_time_last_assigned_map[node_id_w_min_sample] = time_epoch
        return node_id_w_min_sample
