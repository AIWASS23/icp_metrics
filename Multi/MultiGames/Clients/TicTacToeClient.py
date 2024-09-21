
from Identifiers import GameType
import json
from myGame import Game

PLAYER_1 = "PLAYER_1"
PLAYER_2 = "PLAYER_2"

SET_NAME = "SET_NAME"
GET_NAME = "GET_NAME"
WAIT = "WAIT"
START = "START"
PLAY = "PLAY"
UPDATE = "UPDATE"
QUIT = "QUIT"
END = "END"

HOST = '127.0.0.1'
PORT = 5000


class TicTacToeClient:
    def __init__(self):
        self.game = Game()
        self.client_socket = None
        self.type_of_client = GameType.TicTacToe.value

    def get_type_of_client(self):
        return self.type_of_client

    def set_client_socket(self, client_socket):
        self.client_socket = client_socket

    def start_client(self):
        while True:
            data = self.client_socket.recv(2048)

            if not data:
                print("disconnected from server.")
                break

            data = json.loads(data.decode())
            self.game.cells = data["cells"]

            if data["state"] == WAIT:
                self.game.refresh()
                print(data["message"])

            elif data["state"] == START:
                self.game.refresh()
                if data["player"] == PLAYER_1:
                    message = input(data["message"])
                    data = {"cells": self.game.cells, "message": message, "player": data["player"], "state": UPDATE}
                else:
            # starts game for PLAYER_2, print 'waiting your opponent' and wait
                    print(data["message"])
                    data = {"cells": self.game.cells, "message": "PLAYER_2: waiting for PLAYER_1 first move.", "player": data["player"], "state": WAIT}

                json_data = json.dumps(data).encode()
                self.client_socket.send(json_data)

            elif data["state"] == PLAY:
                # update the game
                message = "-1"
                while not (1 <= int(message) <= 10):
                    self.game.refresh()
                    message = input(data["message"])
                    if not message.isnumeric():
                        message = "-1"

                data = {"cells": self.game.cells, "message": message, "player": data["player"], "state": UPDATE}
                json_data = json.dumps(data).encode()
                self.client_socket.send(json_data)

            elif data["state"] == END:
                choice = ""
                while choice != "Y" and choice != "N":
                    self.game.refresh()
                    print(data["message"])
                    choice = str(input("Would you like to play again? (Y/N) : "))
                    choice = choice.upper()

                data = {"cells": self.game.cells, "message": choice, "player": data["player"], "state": END}
                json_data = json.dumps(data).encode()
                self.client_socket.send(json_data)

            elif data["state"] == "QUIT":
                self.game.refresh()
                print(data["message"])
                message = data["player"] + " - notifies server that it's quiting."
                data = {"cells": self.game.cells, "message": message, "player": data["player"], "state": "QUIT"}
                json_data = json.dumps(data).encode()
                self.client_socket.send(json_data)
                break
            else:
                print("the state is : " + data["state"])
                message = input("input: ")
                data["message"] = data["player"] + " - " + message
                json_data = json.dumps(data).encode()
                self.client_socket.send(json_data)

        self.client_socket.close()


if __name__ == '__main__':
    client = TicTacToeClient()
    client.start_client()

