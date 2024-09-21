import socket


class Player:
    def __init__(self):
        self.name = ""
        self.address = ()
        self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.chosen_game = -1  # GameType.NONE.value
        self.type = -1  # PlayerType.NONE.value
        self.state = -1  # PlayerState.NONE.value

    def set_connection(self, connection):
        self.connection = connection

    def get_connection(self):
        return self.connection

    def set_address(self, address):
        self.address = address

    def get_address(self):
        return self.address

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name







