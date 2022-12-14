import abc

from src.sys import node


class SchingAgent(abc.ABC):
    def __init__(self):
        pass

    def __repr__(self):
        return "SchingAgent()"

    @abc.abstractmethod
    def node_id_to_assign(self):
        return None


class SchingAgent_wOnlineLearning(SchingAgent):
    def __init__(self, node_id_list: list[str]):
        self.node_id_list = node_id_list

    def __repr__(self):
        return (
            "SchingAgent_wOnlineLearning( \n"
            f"\t node_id_list= {self.node_id_list} \n"
            ")"
        )

    @abc.abstractmethod
    def record_cost(self, node_id: str, cost: float):
        return None
