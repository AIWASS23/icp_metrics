import enum


class GameType(enum.Enum):
    NONE = -1
    TicTacToe = 1
    GuessingNumber = 2


class PlayerType(enum.Enum):
    NONE = -1
    Player1 = 0
    Player2 = 1


class MessageType(enum.Enum):
    UPDATE_GUI = 0
    MOVE_REQUEST = 1
    END_GAME = 2
    CHECK_ATTEMPTS = 32
    RESPONSE = 4


class ServerState(enum.Enum):
    NONE = -1
    LISTEN = 1
    CONNECTED = 2
    REFRESH = 3


class ServerEvent(enum.Enum):
    WAIT_PLAYER = 1
    START_GAME = 2
    END_GAME = 3
    END_RESET = 4
