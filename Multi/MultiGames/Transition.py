
class Transition:
    def __init__(self, given_state, event, next_state, action):
        self.given_state = given_state
        self.event = event
        self.next_state = next_state
        self.action = action

