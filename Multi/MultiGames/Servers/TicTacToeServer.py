
from Identifiers import GameType
from Identifiers import PlayerType

import time
import threading
import json
from myGame import Game


HOST = '127.0.0.1'
PORT = 5000

SET_NAME = "SET_NAME"
GET_NAME = "GET_NAME"
WAIT = "WAIT"
START = "START"
PLAY = "PLAY"
UPDATE = "UPDATE"
END = "END"
QUIT = "QUIT"


PLAYER_1 = "PLAYER_1"
PLAYER_2 = "PLAYER_2"


class TicTacToeServer:

    def __init__(self):
        self.players = []
        self.game = Game()
        self.mySocket = None
        self.type_of_server = GameType.TicTacToe.value
        self.is_game_ended = False

    def set_client_socket(self, server_socket):
        self.mySocket = server_socket

    def set_players(self, players):
        self.players = players

    def get_type_of_server(self):
        return self.type_of_server

    def get_number_of_players(self):
        return len(self.players)

    def send_request(self, message, player, state):

        if player == "PLAYER_1":
            player_index = int(PlayerType.Player1.value)
        else:
            player_index = int(PlayerType.Player2.value)

        data = {"cells": self.game.cells, "message": message, "player": player, "state": state}
        json_data = json.dumps(data).encode()
        self.players[player_index].get_connection().send(json_data)

    def update_cells(self, player, index):
        if player == PLAYER_1:
            self.game.cells[index] = "X"
        elif player == PLAYER_2:
            self.game.cells[index] = "O"

    def reverse_players_to_send_request(self, player):
        if player == PLAYER_1:
            return PLAYER_2
        elif player == PLAYER_2:
            return PLAYER_1
        else:
            print("something went wrong while reversing players")
            exit(-1)

    def run(self, connection, address):

        while True:
            data = connection.recv(2048)

            if not data:
                break

            data = data.decode()
            data = json.loads(data)

            if data["state"] == UPDATE:
                move = data["message"]
                if not move.isnumeric() or int(move) < 1 or self.game.cells[int(move) - 1] != " ":
                    self.send_request("please enter a number 1-9: ", data["player"], PLAY)

                else:
                    # add new move and update both clients
                    self.update_cells(data["player"], int(move) - 1)
                    if self.game.is_winner(data["player"]):
                        player_index = PlayerType.Player1.value if data["player"] == PLAYER_1 else PlayerType.Player2.value
                        self.send_request(self.players[player_index].get_name().upper() + " WON !!", data["player"],
                                          END)
                        data["player"] = self.reverse_players_to_send_request(data["player"])
                        player_index = PlayerType.Player1.value if data["player"] == PLAYER_1 else PlayerType.Player2.value
                        self.send_request(self.players[player_index].get_name().upper() + " LOST !!", data["player"],
                                          END)

                    elif self.game.is_tie():
                        self.send_request("TIE GAME !!\n", data["player"], END)
                        data["player"] = self.reverse_players_to_send_request(data["player"])
                        self.send_request("TIE GAME !!\n", data["player"], END)

                    else:
                        if len(self.players) != 2:
                            break
                        if self.players[0 if data["player"] == "PLAYER_1" else 1]:
                            data["player"] = self.reverse_players_to_send_request(data["player"])
                            if self.players[0 if data["player"] == "PLAYER_1" else 1]:
                                data["player"] = self.reverse_players_to_send_request(data["player"])
                                self.send_request("waiting for your opponent: ", data["player"], WAIT)  # WAIT
                                data["player"] = self.reverse_players_to_send_request(data["player"])
                                self.send_request("please enter a number 1-9: ", data["player"], PLAY)
                            else:
                                self.game.reset()
                                self.send_request(self.players[0 if data["player"] == "PLAYER_1" else 1].get_name() + " quited the game.\nwaiting for a new player to join...",
                                                  self.reverse_players_to_send_request(data["player"]),
                                                  WAIT)  # WAIT
                        else:
                            data["player"] = self.reverse_players_to_send_request(data["player"])
                            self.game.reset()
                            self.send_request(self.players[0 if data["player"] == "PLAYER_1" else 1].get_name() + " quited the game.\nwaiting for a new player to join...",
                                              self.reverse_players_to_send_request(data["player"]), WAIT)  # WAIT

            elif data["state"] == WAIT:
                print("client send WAIT state to server. message:\n" + data["message"])

            elif data["state"] == END:
                if data["message"] == "Y":
                    if len(self.players) == 2:
                        empty_list = [" ", " ", " ", " ", " ", " ", " ", " ", " "]

                        if self.game.cells != empty_list:

                            if data["player"] == PLAYER_1:
                                self.send_request("waiting for your opponent's answer to play again or not... ",
                                                  PLAYER_1, WAIT)
                                self.game.cells = empty_list
                            else:
                                self.send_request("waiting for your opponent's answer to play again or not... ",
                                                  PLAYER_2, START)
                                self.game.cells = empty_list
                        else:
                            self.send_request("please enter a number 1-9: ", PLAYER_1, START)
                            self.send_request("waiting for your opponent: ", PLAYER_2, START)

                elif data["message"] == "N":

                    if data["player"] == PLAYER_2:
                        if len(self.players) >= 1:
                            self.send_request("TAKE CARE FELLA, SEE YOU LATER !!", PLAYER_2, QUIT)

                        if len(self.players) == 2:
                            message = "Sorry, " + self.players[PlayerType.Player2.value].get_name() + " is quited."
                            self.send_request(message, PLAYER_1, QUIT)
                            #time.sleep(2)
                            self.players.remove(self.players[PlayerType.Player2.value])

                    elif data["player"] == PLAYER_1:
                        if len(self.players) >= 1:
                            self.send_request("TAKE CARE FELLA, SEE YOU LATER !!", PLAYER_1, QUIT)

                        if len(self.players) == 2:
                            message = "Sorry, " + self.players[PlayerType.Player1.value].get_name() + " is quited."
                            self.send_request(message, PLAYER_2, QUIT)
                            #time.sleep(2)
                            self.players.remove(self.players[PlayerType.Player1.value])

            elif data["state"] == QUIT:
                print("client send QUIT state to server. message:\n" + data["message"])

        # WHILE ENDED

        # Disconnection of a client could cause this
        print("Connection Lost..")

        self.game.reset()

        current_thread = threading.current_thread()

        if current_thread.getName() == str(PlayerType.Player1.value):

            print("*\n*\n*\nClient(PLAYER_1) at " + str(address) + " disconnected...\n")
            if len(self.players) >= 1:
                print("player-1 popped")
                if len(self.players) == 2:
                    print("player 2 mesaj gidecek")
                    self.send_request(self.players[PlayerType.Player1.value].get_name() +
                                      " quited the game.", PLAYER_2, QUIT)
                #time.sleep(2)
                self.players.remove(self.players[PlayerType.Player1.value])

        elif current_thread.getName() == str(PlayerType.Player2.value):

            print("*\n*\n*\nClient(PLAYER_2) at " + str(address) + " disconnected...\n")
            if len(self.players) >= 1:
                print("player-2 popped")
                if len(self.players) == 2:
                    print("player 1 mesaj gidecek")
                    self.send_request(self.players[PlayerType.Player2.value].get_name() +
                                      " quited the game.", PLAYER_1, QUIT)

                #time.sleep(2)
                    self.players.remove(self.players[PlayerType.Player2.value])

        else:
            print("thread is trying to disconnect = " + str(current_thread.getName()))

        if len(self.players) == 0:
            print("game is finished.")
            self.is_game_ended = True
        else:
            print("length: " + str(len(self.players)))
            if len(self.players) == 1:
                print(self.players[0].get_name())
            elif len(self.players) == 2:
                print(self.players[0].get_name())
                print(self.players[1].get_name())

    def start_game(self):

        self.is_game_ended = False
        count = PlayerType.Player1.value

        while count <= PlayerType.Player2.value:
            print("thread starting for " + str(self.players[count].get_name()) + " - ("+ str(self.players[count].get_address())+")")

            new_thread = threading.Thread(target=self.run, args=(self.players[count].get_connection(), self.players[count].get_address()))
            new_thread.setName(count)
            print("new thread's name is", str(count))
            new_thread.start()
            count += 1

        self.send_request("waiting for your opponent: ", PLAYER_2, WAIT)
        time.sleep(0.5)
        self.send_request("please enter a number 1-9: ", PLAYER_1, PLAY)

        while not self.is_game_ended:
            time.sleep(1)

        return self.is_game_ended


if __name__ == '__main__':
    self = TicTacToeServer()
    self.start_game()



'''
class ClientThread(threading.Thread):

    def __init__(self, address, client_socket):
        threading.Thread.__init__(self)
        self.address = address
        self.client_connection = client_socket
        print("[+] New thread started for " + str(self.address))

    def run(self):
        while True:
            print(str(self.client_connection)+"\n"+str(self.address))

            data = self.client_connection.recv(2048)

            if not data:
                break

            data = data.decode()
            data = json.loads(data)
            # print("Client(%s:%s) sent : %s" % (self.ip, str(self.port), data))

            if data["state"] == UPDATE:
                move = data["message"]
                if not move.isnumeric() or int(move) < 1 or self.game.cells[int(move) - 1] != " ":
                    self.send_request("please enter a number 1-9: ", data["player"], PLAY)

                else:
                    # add new move and update both clients
                    self.update_cells(data["player"], int(move) - 1)

                    if self.game.is_winner(data["player"]):
                        self.send_request(self.players.get_name.upper() + " WON !!", data["player"],
                                          END)
                        data["player"] = self.reverse_players_to_send_request(data["player"])
                        self.send_request(self.players.get_name.upper() + " LOST !!", data["player"],
                                          END)

                    elif self.game.is_tie():
                        self.send_request("TIE GAME !!\n", data["player"], END)
                        data["player"] = self.reverse_players_to_send_request(data["player"])
                        self.send_request("TIE GAME !!\n", data["player"], END)

                    else:

                        if self.players[data["player"]]:
                            data["player"] = self.reverse_players_to_send_request(data["player"])
                            if self.players[data["player"]]:
                                data["player"] = self.reverse_players_to_send_request(data["player"])
                                self.send_request("waiting for your opponent: ", data["player"], WAIT)  # WAIT
                                data["player"] = self.reverse_players_to_send_request(data["player"])
                                self.send_request("please enter a number 1-9: ", data["player"], PLAY)
                            else:
                                self.game.reset()
                                self.send_request(self.players[data["player"]][
                                                        "name"] + " quited the game.\nwaiting for a new player to join...",
                                                  self.reverse_players_to_send_request(data["player"]),
                                                  WAIT)  # WAIT
                        else:
                            data["player"] = self.reverse_players_to_send_request(data["player"])
                            self.game.reset()
                            self.send_request(self.players[data["player"]][
                                                    "name"] + " quited the game.\nwaiting for a new player to join...",
                                              self.reverse_players_to_send_request(data["player"]), WAIT)  # WAIT

            elif data["state"] == WAIT:
                print("client send WAIT state to server. message:\n" + data["message"])

            elif data["state"] == END:
                if data["message"] == "Y":
                    if self.players[PLAYER_1] and self.players[PLAYER_2]:
                        # reset cells
                        empty_list = [" ", " ", " ", " ", " ", " ", " ", " ", " "]

                        if self.game.cells != empty_list:

                            if data["player"] == PLAYER_1:
                                self.send_request("waiting for your opponent's answer to play again or not... ",
                                                  PLAYER_1, WAIT)
                                self.game.cells = empty_list
                            else:
                                # player 2 first said yes
                                self.send_request("waiting for your opponent's answer to play again or not... ",
                                                  PLAYER_2, START)
                                self.game.cells = empty_list
                        else:
                            self.send_request("please enter a number 1-9: ", PLAYER_1, START)
                            self.send_request("waiting for your opponent: ", PLAYER_2, START)

                elif data["message"] == "N":
                    if self.players[PLAYER_1]:
                        self.send_request("Thanks for Playing !! ", PLAYER_1, QUIT)
                    if self.players[PLAYER_2]:
                        self.send_request("Thanks for Playing !! ", PLAYER_2, QUIT)

            elif data["state"] == QUIT:
                self.players[data["player"]] = {}
                print("client send QUIT state to server. message:\n" + data["message"])

        # WHILE ENDED

        # Disconnection of a client could cause this

        self.game.reset()

        if self.players[PLAYER_1] and self.port == self.players[PLAYER_1]["connection"]["port"]:
            if self.players[PLAYER_2]:
                self.send_request(
                    self.players[PLAYER_1]["name"] + " quited the game.\nwaiting for a new player to join...",
                    PLAYER_2, "WAIT")
            print("*\n*\n*\nClient(PLAYER_1) at " + str(self.ip) + ":" + str(self.port) + " disconnected...\n")
            self.players[PLAYER_1] = {}

        elif self.players[PLAYER_2] and self.port == self.players[PLAYER_2]["connection"]["port"]:
            if self.players[PLAYER_1]:
                self.send_request(
                    self.players[PLAYER_2]["name"] + " quited the game.\nwaiting for a new player to join...",
                    PLAYER_1, "WAIT")
            print("Client(PLAYER_2) at " + str(self.ip) + ":" + str(self.port) + " disconnected...\n")
            self.players[PLAYER_2] = {}



class TicTacToeServer:

    def __init__(self):
        self.players = []
        self.game = Game()
        self.mySocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.mySocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.mySocket.bind((HOST, PORT))
        self.connection = None
        self.type_of_server = GameType.TicTacToe.value

    def set_players(self, players):
        self.players = players

    def get_type_of_server(self):
        return self.type_of_server

    def get_number_of_players(self):
        return len(self.players)

    def send_request(self, message, player, state):
        player_index = -1
        if player == "PLAYER_1":
            player_index = int(PlayerType.Player1.value)
        else:
            player_index = int(PlayerType.Player2.value)

        data = {"cells": self.game.cells, "message": message, "player": player, "state": state}
        json_data = json.dumps(data).encode()
        self.players[player_index].get_connection().send(json_data)
        # self.players[player]["connection"]["socket"].send(json_data)



    def update_cells(self, player, index):
        if player == PLAYER_1:
            self.game.cells[index] = "X"
        elif player == PLAYER_2:
            self.game.cells[index] = "O"


    def reverse_players_to_send_request(self, player):
        if player == PLAYER_1:
            return PLAYER_2
        elif player == PLAYER_2:
            return PLAYER_1
        else:
            print("something went wrong while reversing players")
            exit(-1)


    def start_game(self):
        # print("Connection from : " + self.ip + ":" + str(PORT))
        message = "Welcome to the server.\n"


        count = PlayerType.Player1.value

        while count <= PlayerType.Player2.value:
            print("thread starting for "+ str(self.players[count].get_name()) + " - ("+ str(self.players[count].get_address())+")")
            new_thread = ClientThread(self.players[count].get_address(), self.players[count].get_connection())
            new_thread.start()
            count += 1



        self.send_request("waiting for your opponent: ", PLAYER_2, START)
        time.sleep(0.5)
        self.send_request("please enter a number 1-9: ", PLAYER_1, START)



if __name__ == '__main__':
    self = TicTacToeServer()
    self.start_game()

'''
