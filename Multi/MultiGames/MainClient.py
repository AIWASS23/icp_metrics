import socket
import os
from Clients.TicTacToeClient import TicTacToeClient
from Clients.GuessingNumberClient import GuessingNumberClient
from Identifiers import GameType
# from TicTacToe.Tic_tac_toe_game import start_Tic_Tac_Toe_game
# from GuessNumber.GuessNumber import start_guess_the_number


HOST = '127.0.0.1'
PORT = 9999


class MainClient:
    def __init__(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((HOST, PORT))
        self.list_of_clients = [TicTacToeClient(), GuessingNumberClient()]

    def start_client(self):
        print("waiting for server connection...")

        # connection establishment
        name_asked = self.client_socket.recv(1024)
        if not name_asked:
            print("receiving name asking is failed")
            exit(-1)

        # insert name
        name_asked = str(input(name_asked.decode()))
        self.client_socket.send(name_asked.encode())

        os.system("clear")
        print("\n\n")
        # request for a game mode
        request_game_mode = self.client_socket.recv(1024)
        if not request_game_mode:
            print("first connection could not be established with the server.")
            exit(-1)

        print(request_game_mode.decode())

        # insert game mode
        while True:
            game_mode = str(input())
            os.system("clear")
            print("\n\n")

            if game_mode in [str(GameType.TicTacToe.value), str(GameType.GuessingNumber.value)]:
                self.client_socket.send(game_mode.encode())
                waiting_message = self.client_socket.recv(1024)
                print(waiting_message.decode())
                break
            else:
                print("you entered a wrong choice\nPlease choose a game :\n1 for Tic Tac Toe \n2 for Guess a number \n")

        for game_client in self.list_of_clients:
            if game_client.get_type_of_client() == int(game_mode):
                game_client.set_client_socket(self.client_socket)
                game_client.start_client()



MainClient().start_client()