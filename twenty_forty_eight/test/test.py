"""test.py -- Tests the game."""

from twenty_forty_eight.game import Game


if __name__ == "__main__":
    game = Game(
        test_input=[
            [0, 2, 0, 0],
            [2, 0, 0, 0],
            [2, 2, 0, 0],
            [2, 4, 0, 0],
            [2, 4, 4, 0],
            [2, 4, 4, 2],
            [0, 2, 2, 0],
            [0, 0, 0, 0],
            [2, 0, 2, 0],
            [2, 2, 2, 0],
            [2, 2, 2, 2],
            [2, 0, 0, 2],
            [2, 0, 0, 4],
        ],
    )
    game.move_left()
    EXPECTED_GAME_STATE = """
|---------+---------+---------+---------|
|         |         |         |         |
|    2    |         |         |         |
|         |         |         |         |
|---------+---------+---------+---------|
|         |         |         |         |
|    2    |         |         |         |
|         |         |         |         |
|---------+---------+---------+---------|
|         |         |         |         |
|    4    |         |         |         |
|         |         |         |         |
|---------+---------+---------+---------|
|         |         |         |         |
|    2    |    4    |         |         |
|         |         |         |         |
|---------+---------+---------+---------|
|         |         |         |         |
|    2    |    8    |         |         |
|         |         |         |         |
|---------+---------+---------+---------|
|         |         |         |         |
|    2    |    8    |    2    |         |
|         |         |         |         |
|---------+---------+---------+---------|
|         |         |         |         |
|    4    |         |         |         |
|         |         |         |         |
|---------+---------+---------+---------|
|         |         |         |         |
|         |         |         |         |
|         |         |         |         |
|---------+---------+---------+---------|
|         |         |         |         |
|    4    |         |         |         |
|         |         |         |         |
|---------+---------+---------+---------|
|         |         |         |         |
|    4    |    2    |         |         |
|         |         |         |         |
|---------+---------+---------+---------|
|         |         |         |         |
|    4    |    4    |         |         |
|         |         |         |         |
|---------+---------+---------+---------|
|         |         |         |         |
|    4    |         |         |         |
|         |         |         |         |
|---------+---------+---------+---------|
|         |         |         |         |
|    2    |    4    |         |         |
|         |         |         |         |
|---------+---------+---------+---------|
"""

    if str(game).strip() == EXPECTED_GAME_STATE.strip():
        print("Tests pass!")
    else:
        print("Tests failed, run in debug mode!")
        print(game, EXPECTED_GAME_STATE)
