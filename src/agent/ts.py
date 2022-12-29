import collections
import numpy

from typing import Tuple

from src.agent import agent, exp as exp_module
from src.prob import random_variable
from src.sys import node
from src.utils.debug import *


class AssignWithThompsonSampling_slidingWin(agent.SchingAgent_wOnlineLearning):
    def __init__(self, node_list: list[node.Node], win_len: int):
        super().__init__(node_list=node_list)
        self.win_len = win_len

        self.node_id_and_exp_queue = collections.deque(maxlen=win_len)

    def __repr__(self):
        return (
            "AssignWithThompsonSampling_slidingWin( \n"
            f"\t node_id_list= {self.node_id_list} \n"
            f"\t win_len= {self.win_len} \n"
            ")"
        )

    def record_exp(self, node_id: str, exp: exp_module.Exp):
        self.node_id_and_exp_queue.append((node_id, exp))
        log(DEBUG, "recorded", node_id=node_id, exp=exp)

    def node_id_to_assign(self, time_epoch: float=None):
        # Construct `node_id_to_wait_times_map`
        node_id_to_wait_times_map = collections.defaultdict(list)
        for (node_id, exp) in self.node_id_and_exp_queue:
            node_id_to_wait_times_map[node_id].append(exp.wait_time)

        for node_id in self.node_id_list:
            if node_id not in node_id_to_wait_times_map:
                node_id_to_wait_times_map[node_id].append(0)

        log(DEBUG, "", node_id_to_wait_times_map=node_id_to_wait_times_map)

        # Choose the node with min wait time sample
        node_id_to_return, min_sample = None, float("Inf")
        for node_id, wait_times in node_id_to_wait_times_map.items():
            mean = numpy.mean(wait_times) if len(wait_times) else 0
            stdev = numpy.std(wait_times) if len(wait_times) else 1
            check(stdev >= 0, "Stdev cannot be negative")
            if stdev == 0:
                stdev = 1

            s = random_variable.TruncatedNormal(mu=mean, sigma=stdev).sample()
            if s < min_sample:
                min_sample = s
                node_id_to_return = node_id
                # log(DEBUG, "s < min_sample", s=s, min_sample=min_sample, node_id_to_return=node_id_to_return)

        return node_id_to_return


class AssignWithThompsonSampling_slidingWinForEachNode(agent.SchingAgent_wOnlineLearning):
    def __init__(self, node_list: list[node.Node], win_len: int):
        super().__init__(node_list=node_list)
        self.win_len = win_len

        self.node_id_to_exp_queue_map = {node_id: collections.deque(maxlen=win_len) for node_id in self.node_id_list}

    def __repr__(self):
        return (
            "AssignWithThompsonSampling_slidingWinForEachNode( \n"
            f"\t node_id_list= {self.node_id_list} \n"
            f"\t win_len= {self.win_len} \n"
            ")"
        )

    def mean_stdev_wait_time(self, node_id: str) -> Tuple[float, float]:
        wait_times = [exp.wait_time for exp in self.node_id_to_exp_queue_map[node_id]]
        mean = numpy.mean(wait_times) if len(wait_times) else 0
        stdev = numpy.std(wait_times) if len(wait_times) else 0.01
        check(stdev >= 0, "Stdev cannot be negative")
        if stdev == 0:
            stdev = 0.01

        return mean, stdev

    def record_exp(self, node_id: str, exp: exp_module.Exp):
        self.node_id_to_exp_queue_map[node_id].append(exp)
        log(DEBUG, "recorded", node_id=node_id, exp=exp)

    def node_id_to_assign(self, time_epoch: float=None):
        log(DEBUG, "", node_id_to_exp_queue_map=self.node_id_to_exp_queue_map)

        # Choose the node with min wait time sample
        node_id_to_return, min_sample = None, float("Inf")
        for node_id in self.node_id_to_exp_queue_map:
            mean, stdev = self.mean_stdev_wait_time(node_id)

            s = random_variable.TruncatedNormal(mu=mean, sigma=stdev).sample()
            if s < min_sample:
                min_sample = s
                node_id_to_return = node_id
                # log(DEBUG, "s < min_sample", s=s, min_sample=min_sample, node_id_to_return=node_id_to_return)

        return node_id_to_return


class AssignWithThompsonSampling_resetWinOnRareEvent(AssignWithThompsonSampling_slidingWinForEachNode):
    def __init__(self, node_list: list[node.Node], win_len: int, threshold_prob_rare: float):
        super().__init__(node_list=node_list, win_len=win_len)
        self.threshold_prob_rare = threshold_prob_rare

        self.node_id_to_time_last_assigned_map = {node_id: 0 for node_id in self.node_id_list}

    def __repr__(self):
        return (
            "AssignWithThompsonSampling_ResetWinOnRareEvent( \n"
            f"\t node_id_list= {self.node_id_list} \n"
            f"\t win_len= {self.win_len} \n"
            f"\t threshold_prob_rare= {self.threshold_prob_rare} \n"
            ")"
        )

    def record_exp(self, node_id: str, exp: exp_module.Exp):
        def record():
            self.node_id_to_exp_queue_map[node_id].append(exp)
            log(DEBUG, "recorded", node_id=node_id, exp=exp)

        if len(self.node_id_to_exp_queue_map[node_id]) < 5:
            record()

        else:
            mean, stdev = self.mean_stdev_wait_time(node_id)
            cost_rv = random_variable.TruncatedNormal(mu=mean, sigma=stdev)

            Pr_getting_larger_than_cost = cost_rv.tail_prob(cost)
            Pr_getting_smaller_than_cost = cost_rv.cdf(cost)
            Pr_cost_is_rare = 1 - min(Pr_getting_larger_than_cost, Pr_getting_smaller_than_cost)
            if Pr_cost_is_rare >= self.threshold_prob_rare:
                log(DEBUG, "Rare event detected", cost=cost, mean=mean, stdev=stdev, Pr_cost_is_rare=Pr_cost_is_rare, threshold_prob_rare=self.threshold_prob_rare)
                self.node_id_to_exp_queue_map[node_id].clear()
            else:
                record()

    def node_id_to_assign(self, time_epoch: float):
        log(DEBUG, "",
            node_id_to_exp_queue_map=self.node_id_to_exp_queue_map,
            node_list=[node.repr_w_state() for node in self.node_list],
            time_epoch=time_epoch,
        )

        # Choose the node with min-cost sample
        node_id_to_return, min_sample = None, float("Inf")
        for node_id in self.node_id_to_exp_queue_map:
            time_last_assigned = self.node_id_to_time_last_assigned_map[node_id]

            _mean, _stdev = self.mean_stdev_wait_time(node_id)
            mean = _mean - (time_epoch - self.node_id_to_time_last_assigned_map[node_id])
            if mean <= 0:
                log(DEBUG, "Mean < 0, resetting memory buffer", node_id=node_id)
                self.node_id_to_exp_queue_map[node_id].clear()
                s = mean
            else:
                stdev = _stdev * (1 - mean / _mean)
                s = random_variable.TruncatedNormal(mu=mean, sigma=stdev).sample()

            if s < min_sample:
                min_sample = s
                node_id_to_return = node_id
                # log(DEBUG, "s < min_sample", s=s, min_sample=min_sample, node_id_to_return=node_id_to_return)

        self.node_id_to_time_last_assigned_map[node_id_to_return] = time_epoch
        return node_id_to_return
