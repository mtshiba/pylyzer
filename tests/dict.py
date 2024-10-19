dic = {"a": 1, "b": 2}

def f():
    dic = {"a": 1}
    _ = dic["b"] # ERR


class TaskManager:
    def __init__(self):
        self.tasks: list[dict[str, int]] = []

    def add_task(self, title: str, id: int):
        task = {title: id}
        self.tasks.append(task)

    def add_task2(self, title: str, id: int):
        task = {id: title}
        self.tasks.append(task) # ERR
