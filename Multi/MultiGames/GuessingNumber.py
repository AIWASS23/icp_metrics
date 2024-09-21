import random
from Player import Player

player = Player()


def generate_number():
    winner_number = random.randint(1, 100)
    return winner_number


def client_number_is_too_low(number):
    number = str(number)
    number = "\n"+"your guess : " + number + " is too LOW (COLD)\n"
    return number


def client_number_is_too_high(number):
    number = str(number)
    number = "\n" + "your guess : " + number + " is too HIGH (HOT)\n"
    return number


def choice_number_win(number, name):
    number = str(number)
    you_win = "\n" + name.upper() + " with number : " + number + " YOU WON !!\n"
    return you_win


def you_lose_match(name):
    you_lose = "\n" + name.upper() + " YOU LOST !!\n"
    return you_lose
