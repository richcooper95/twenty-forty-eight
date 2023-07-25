# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
import numpy as np

from twenty_forty_eight.game import Game


class TestGame:
    def test_init(self):
        game = Game(
            num_rows=2,
            num_cols=4,
            up_key="w",
            down_key="s",
            left_key="a",
            right_key="d",
        )
        np.testing.assert_array_equal(game.mat, np.zeros((2, 4)))
        assert game.debug is False
        assert game.up_key == "w"
        assert game.down_key == "s"
        assert game.left_key == "a"
        assert game.right_key == "d"

    def test_init_debug(self):
        game = game = Game(
            num_rows=2,
            num_cols=4,
            up_key="w",
            down_key="s",
            left_key="a",
            right_key="d",
            debug=True,
        )
        np.testing.assert_array_equal(game.mat, np.zeros((2, 4)))
        assert game.debug is True
        assert game.up_key == "w"
        assert game.down_key == "s"
        assert game.left_key == "a"
        assert game.right_key == "d"

    def test_str(self):
        game = Game(
            num_rows=2,
            num_cols=4,
            up_key="w",
            down_key="s",
            left_key="a",
            right_key="d",
        )

        expected_output = """
|---------+---------+---------+---------|
|         |         |         |         |
|         |         |         |         |
|         |         |         |         |
|---------+---------+---------+---------|
|         |         |         |         |
|         |         |         |         |
|         |         |         |         |
|---------+---------+---------+---------|
"""

        assert str(game).strip() == expected_output.strip()

    def test_str_with_values(self):
        game = Game(
            num_rows=2,
            num_cols=4,
            up_key="w",
            down_key="s",
            left_key="a",
            right_key="d",
        )

        game.mat[0, 0] = 2
        game.mat[0, 1] = 4
        game.mat[0, 2] = 8
        game.mat[0, 3] = 16
        game.mat[1, 0] = 32
        game.mat[1, 1] = 64
        game.mat[1, 2] = 128
        game.mat[1, 3] = 256

        expected_output = """
|---------+---------+---------+---------|
|         |         |         |         |
|    2    |    4    |    8    |   16    |
|         |         |         |         |
|---------+---------+---------+---------|
|         |         |         |         |
|   32    |   64    |   128   |   256   |
|         |         |         |         |
|---------+---------+---------+---------|
"""

        assert str(game).strip() == expected_output.strip()

    def test_num_rows(self):
        game = Game(
            num_rows=2,
            num_cols=4,
            up_key="w",
            down_key="s",
            left_key="a",
            right_key="d",
        )
        assert game.num_rows == 2

    def test_num_cols(self):
        game = Game(
            num_rows=2,
            num_cols=4,
            up_key="w",
            down_key="s",
            left_key="a",
            right_key="d",
        )
        assert game.num_cols == 4

    def test_format_number(self):
        assert Game.format_number(0) == " "
        assert Game.format_number(2) == "2"

    def test_no_moves_available(self):
        game = Game(
            num_rows=2,
            num_cols=4,
            up_key="w",
            down_key="s",
            left_key="a",
            right_key="d",
        )

        game.mat[0, 0] = 2
        game.mat[0, 1] = 4
        game.mat[0, 2] = 8
        game.mat[0, 3] = 16
        game.mat[1, 0] = 32
        game.mat[1, 1] = 64
        game.mat[1, 2] = 128
        game.mat[1, 3] = 256

        assert game.no_moves_available() is True

    def test_no_moves_available_with_empty_cell(self):
        game = Game(
            num_rows=2,
            num_cols=4,
            up_key="w",
            down_key="s",
            left_key="a",
            right_key="d",
        )

        game.mat[0, 0] = 2
        game.mat[0, 1] = 4
        game.mat[0, 2] = 8
        game.mat[0, 3] = 16
        game.mat[1, 0] = 32
        game.mat[1, 1] = 64
        game.mat[1, 2] = 128

        assert game.no_moves_available() is False

    def test_no_moves_available_with_horizontal_merge(self):
        game = Game(
            num_rows=2,
            num_cols=4,
            up_key="w",
            down_key="s",
            left_key="a",
            right_key="d",
        )

        game.mat[0, 0] = 2
        game.mat[0, 1] = 4
        game.mat[0, 2] = 8
        game.mat[0, 3] = 16
        game.mat[1, 0] = 32
        game.mat[1, 1] = 64
        game.mat[1, 2] = 128
        game.mat[1, 3] = 128

        assert game.no_moves_available() is False

    def test_no_moves_available_with_vertical_merge(self):
        game = Game(
            num_rows=2,
            num_cols=4,
            up_key="w",
            down_key="s",
            left_key="a",
            right_key="d",
        )

        game.mat[0, 0] = 2
        game.mat[0, 1] = 4
        game.mat[0, 2] = 8
        game.mat[0, 3] = 256
        game.mat[1, 0] = 32
        game.mat[1, 1] = 64
        game.mat[1, 2] = 128
        game.mat[1, 3] = 256

        assert game.no_moves_available() is False
