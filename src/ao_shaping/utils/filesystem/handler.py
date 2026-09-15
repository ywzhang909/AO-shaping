class Register:
    def __init__(self) -> None:
        self.members = {}

    def register(self, name: str):
        def decorator(func):
            self.members[name] = func
            return func

        return decorator

    def __getitem__(self, name: str):
        return self.members[name]

    @property
    def all_funcs(self):
        return self.members.values()

    @property
    def all_names(self):
        return self.members.keys()
