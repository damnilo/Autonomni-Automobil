from dataclasses import dataclass

@dataclass
class Transition:

    state: any

    action: int

    reward: float

    done: bool

    next_states: any