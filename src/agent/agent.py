import abc


class SchingAgent(abc.ABC):
    def __init__(self, node_id_list: list[str]):
        self.node_id_list = node_id_list

    def __repr__(self):
        return (
            f"SchingAgent( \n"
            f"\t node_id_list= {self.node_id_list} \n"
            ")"
        )

    @abc.abstractmethod
    def record_cost(self, node_id: str, cost: float):
        return None

    @abc.abstractmethod
    def node_to_schedule(self) -> str:
        return None
