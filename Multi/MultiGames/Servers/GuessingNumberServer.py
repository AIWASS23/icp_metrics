from Identifiers import *
import json
from GuessingNumber import *


class GuessingNumberServer:
    def __init__(self):
        print("GuessingNumberServer server initialized")
        self.type_of_server = GameType.GuessingNumber.value
        self.players = []
        self.player_winner = ""
        self.winner_number = 0
        self.number_to_guess = generate_number()
        self.is_game_ended = False
        self.number_of_attempts = [0, 0]
        self.type_of_server = GameType.GuessingNumber.value

    def get_type_of_server(self):
        return self.type_of_server

    def add_player(self, player):
        self.players.append(player)

    def set_players(self, players):
        self.players = players

    def get_number_of_players(self):
        return len(self.players)

    def send_request(self, message, player, type):
        json_data = json.dumps(
            {"message": message, "name": self.players[player].get_name(), "type": type})
        self.players[player].get_connection().send(json_data.encode())

    def check_attempts(self, number_of_attempts, player):

        request_message = "\nplease choose a number between 1 to 100 --> "

        if 4 < number_of_attempts < 7:
            message = "a few attempts left be careful." + request_message
        elif number_of_attempts == 7:
            message = " this is your last chance... Use it wisely !!" + request_message
        else:
            message = request_message

        self.send_request(message, player, MessageType.CHECK_ATTEMPTS.value)

    def main_logic(self, number, player):
        if number > self.number_to_guess:
            self.send_request(client_number_is_too_high(number), player, MessageType.RESPONSE.value)
            return False

        elif number < self.number_to_guess:
            self.send_request(client_number_is_too_low(number), player, MessageType.RESPONSE.value)
            return False

        else:
            self.player_winner = self.players[player].get_name()
            self.winner_number = number
            return True

    def start_game(self):

        print("Guessing Number Server - start_game is running.")
        self.is_game_ended = False

        welcome_message = "*** STARTING MATCH ***\n" + \
                          self.players[PlayerType.Player1.value].get_name() + " vs " + self.players[
                              PlayerType.Player2.value].get_name() + "\nGuessing a number challenge:"

        print(welcome_message)

        self.players[PlayerType.Player1.value].get_connection().send(welcome_message.encode())
        self.players[PlayerType.Player2.value].get_connection().send(welcome_message.encode())

        # time.sleep(2)

        print(self.number_to_guess)

        while not self.is_game_ended and (self.number_of_attempts[PlayerType.Player1.value] +
                                     self.number_of_attempts[PlayerType.Player2.value]) < 16:

            if self.number_of_attempts[PlayerType.Player1.value] <= 8:
                self.check_attempts(self.number_of_attempts[PlayerType.Player1.value], PlayerType.Player1.value)

                data = self.players[PlayerType.Player1.value].get_connection().recv(1024)
                if not data:
                    print("receiving 'guess_number_of_player1' failed.")
                    self.send_request(self.players[PlayerType.Player1.value].get_name() + " is quited.",
                                      PlayerType.Player2.value, MessageType.END_GAME.value)
                    self.is_game_ended = True
                    break

                data = data.decode()
                guess_number_of_player1 = int(data)

                self.is_game_ended = self.main_logic(guess_number_of_player1, PlayerType.Player1.value)

                self.number_of_attempts[PlayerType.Player1.value] += 1

            if not self.is_game_ended and self.number_of_attempts[PlayerType.Player2.value] <= 8:
                self.check_attempts(self.number_of_attempts[PlayerType.Player2.value], PlayerType.Player2.value)

                data = self.players[PlayerType.Player2.value].get_connection().recv(1024)
                if not data:
                    print("receiving 'guess_number_of_player2' failed.")
                    self.send_request(self.players[PlayerType.Player2.value].get_name() + " is quited.",
                                      PlayerType.Player1.value, MessageType.END_GAME.value)
                    self.is_game_ended = True
                    break

                data = data.decode()
                number_player2 = int(data)

                self.is_game_ended = self.main_logic(number_player2, PlayerType.Player2.value)

                self.number_of_attempts[PlayerType.Player2.value] += 1

        if self.is_game_ended:

            if self.player_winner == self.players[PlayerType.Player1.value].get_name():
                self.send_request(
                    choice_number_win(self.winner_number, self.players[PlayerType.Player1.value].get_name()),
                    PlayerType.Player1.value, MessageType.END_GAME.value)
                self.send_request(you_lose_match(self.players[PlayerType.Player2.value].get_name()),
                                  PlayerType.Player2.value, MessageType.END_GAME.value)
            elif self.player_winner == self.players[PlayerType.Player2.value].get_name():
                self.send_request(
                    choice_number_win(self.winner_number, self.players[PlayerType.Player2.value].get_name()),
                    PlayerType.Player2.value, MessageType.END_GAME.value)
                self.send_request(you_lose_match(self.players[PlayerType.Player1.value].get_name()),
                                  PlayerType.Player1.value, MessageType.END_GAME.value)
        else:
            self.send_request(you_lose_match(self.players[PlayerType.Player1.value].get_name()),
                              PlayerType.Player1.value, MessageType.END_GAME.value)
            self.send_request(you_lose_match(self.players[PlayerType.Player2.value].get_name()),
                              PlayerType.Player2.value, MessageType.END_GAME.value)

        return True
