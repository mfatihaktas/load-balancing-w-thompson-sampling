import abc
import numpy

from src.agent import agent


class ThompsonSampling_slidingWin_Gaussian(agent.SchingAgent):
    def __init__(self, node_id_list, win_len):
        super().__init__(node_id_list=node_id_list)

        self.win_len = win_len

        self.node_id_cost_queue = deque(maxlen=win_len)
        for node_id in self.node_id_list:
            self.node_id_cost_queue.append((node_id, 0))

    def __repr__(self):
        return (
            "ThompsonSampling_slidingWin_Gaussian( \n"
            f"{super().__repr__()} \n"
            f"\t win_len= {self.win_len} \n"
            ")"
        )

    def record_cost(self, node_id: str, cost: float):
        self.node_id_cost_queue.append((node_id, cost))
        log(DEBUG, "recorded", node_id=node_id, cost=cost)

    def node_to_schedule(self) -> str:
        node_id_to_costs_map = defaultdict(list)
        for (node_id, cost) in self.node_id_cost_queue:
            node_id_to_costs_map[node_id].append(cost)
        log(DEBUG, "", node_id_to_costs_map=node_id_to_costs_map)

        node_id, min_sample = None, float("Inf")
        for node_id, cost_list in node_id_to_costs_map.items():
            mean = np.mean(cost_list) if len(cost_list) else 0
            stdev = np.std(cost_list) if len(cost_list) else 1
            check(stdev >= 0, "Stdev cannot be negative")
            if stdev == 0:
                stdev = 1

            s = numpy.random.normal(loc=mean, scale=stdev)
            if s < min_sample:
                min_sample = s
                node_id = node_id
                # log(DEBUG, "s < min_sample", s=s, min_sample=min_sample, node_id=node_id)

        return node_id
