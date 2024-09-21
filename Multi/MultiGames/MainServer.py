import socket
from Transition import Transition
from Identifiers import *
from Player import Player
from Servers.TicTacToeServer import TicTacToeServer
from Servers.GuessingNumberServer import GuessingNumberServer
import time

HOST = '127.0.0.1'
PORT = 9999


class MainServer:
    def __init__(self):
        self.players = []
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((HOST, PORT))
        self.server_socket.listen(5)
        self.server_current_state = ServerState.LISTEN

        self.list_of_servers = [TicTacToeServer(), GuessingNumberServer()]
        self.game_type = GameType.NONE.value
        self.is_game_ended = False

        self.transitions = [
            Transition(ServerState.LISTEN, ServerEvent.WAIT_PLAYER, ServerState.LISTEN, self.start_server),
            Transition(ServerState.LISTEN, ServerEvent.START_GAME, ServerState.CONNECTED, self.start_game),
            Transition(ServerState.CONNECTED, ServerEvent.END_GAME, ServerState.REFRESH, self.reset_game),
            Transition(ServerState.REFRESH, ServerEvent.END_RESET, ServerState.LISTEN, self.start_game)  # self.start_server
        ]

    def handle_server_state_transitions(self, event):

        print("As design pattern FSM transition table is used to handle server states.")
        for transition in self.transitions:
            if transition.event == event and self.server_current_state == transition.given_state:
                print("Current State \t:\t", self.server_current_state)
                self.server_current_state = transition.next_state
                print("Next State \t:\t", self.server_current_state, "\n")
                transition.action()

    def reset_game(self):
        self.players.clear()
        self.game_type = GameType.NONE.value
        self.is_game_ended = False
        self.handle_server_state_transitions(ServerEvent.END_RESET)

    def start_game(self):

        for game_server in self.list_of_servers:
            if game_server.get_type_of_server() == int(self.game_type):
                if game_server.get_type_of_server() == GameType.TicTacToe.value \
                        or game_server.get_type_of_server() == GameType.GuessingNumber.value:

                    game_server.set_players(self.players)
                    print("game started")
                    self.is_game_ended = game_server.start_game()
                    print("game ended")
                else:
                    print("error :: unexpected game type !!")
                    self.handle_server_state_transitions(ServerEvent.END_RESET)

        if self.is_game_ended:
            self.handle_server_state_transitions(ServerEvent.END_GAME)

    def start_server(self):

        print("Server listening.")
        print(HOST)

        while True:

            connection, address = self.server_socket.accept()

            print('[+] New connection from', str(address[0]), str(address[1]))

            player = Player()
            player.set_address(address)
            player.set_connection(connection)

            # self.players[PlayerType.Player1.value] = player
            message = "Welcome " + str(player.get_address()) + "\nPlease enter your name: "
            player.get_connection().send(message.encode())

            message = player.get_connection().recv(1024).decode()

            if not message:
                print("Player may be disconnected. (while entering name)")
                player.get_connection().close()
                continue

            player.set_name(message)
            print("(", str(address[0]), str(address[1]), ") player's name is", player.get_name())

            if len(self.players) >= 1:

                message = "At the moment the server is hosting "
                game_message = ""

                if self.game_type == GameType.TicTacToe.value:
                    game_message = "the tic tac toe game "

                elif self.game_type == GameType.GuessingNumber.value:
                    game_message = "guessing a number game "

                player.get_connection().send(
                    (message + game_message + "\nIf you want to join " + self.players[
                        PlayerType.Player1.value].get_name() + " insert " + str(self.game_type)).encode())

                data = player.get_connection().recv(1024).decode()

                if not data:
                    print("Lost connection - while choosing game mode")
                    player.get_connection().close()
                    continue

                game_mode_choice = int(data)

                if game_mode_choice == self.game_type:
                    player.get_connection().send(("Prepare for the match " + str(player.get_name())).encode())
                    self.players.append(player)
                else:
                    player.get_connection().close()

            else:
                message = "Please choose a game :\n1 for Tic Tac Toe \n2 for Guess a number \n "
                player.get_connection().send(message.encode())

                data = player.get_connection().recv(1024).decode()

                if not data:
                    print("Lost connection - while choosing game mode")
                    player.get_connection().close()
                    #if len(self.players) > 0:
                     #   self.players.pop()
                    continue

                self.game_type = int(data)

                player.get_connection().send(
                    ("Prepare for the match " + str(player.get_name()) + "\nWaiting for opponents").encode())

                self.players.append(player)

            if len(self.players) == 2:
                self.handle_server_state_transitions(ServerEvent.START_GAME)


if __name__ == '__main__':
    server = MainServer()
    server.handle_server_state_transitions(ServerEvent.WAIT_PLAYER)
