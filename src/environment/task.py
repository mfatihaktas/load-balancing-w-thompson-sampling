class Task:
    def __init__(
        self,
        _id: str,
    ):
        self._id = _id

    def __repr__(self):
        # return (
        #     "Msg( \n"
        #     f"\t id= {self._id} \n"
        #     ")"
        # )

        return f"Message(id= {self._id})"
