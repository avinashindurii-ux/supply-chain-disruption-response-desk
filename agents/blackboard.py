"""Shared disruption-state board: every agent reads and writes here."""


class Blackboard:
    def __init__(self):
        self.state = {}
        self.log = []          # (agent, key) audit trail

    def write(self, agent, key, value):
        self.state[key] = value
        self.log.append((agent, key))

    def read(self, key, default=None):
        return self.state.get(key, default)
