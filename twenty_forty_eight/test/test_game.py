# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
import pytest
import unittest.mock as mock

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

    def test_is_full(self):
        game = Game(
            num_rows=2,
            num_cols=2,
            up_key="w",
            down_key="s",
            left_key="a",
            right_key="d",
        )

        game.mat[0, 0] = 2
        game.mat[0, 1] = 4
        game.mat[1, 0] = 8
        game.mat[1, 1] = 16

        assert game.is_full() is True

    def test_is_full_with_empty_cell(self):
        game = Game(
            num_rows=2,
            num_cols=2,
            up_key="w",
            down_key="s",
            left_key="a",
            right_key="d",
        )

        game.mat[0, 0] = 2
        game.mat[0, 1] = 4
        game.mat[1, 0] = 8

        assert game.is_full() is False

    def test_game_won_with_2048(self):
        game = Game(
            num_rows=2,
            num_cols=2,
            up_key="w",
            down_key="s",
            left_key="a",
            right_key="d",
        )

        game.max_val = 2048

        assert game.game_won() is True

    def test_game_won_without_2048(self):
        game = Game(
            num_rows=2,
            num_cols=2,
            up_key="w",
            down_key="s",
            left_key="a",
            right_key="d",
        )

        game.max_val = 1024

        assert game.game_won() is False

    @mock.patch("random.randint", return_value=1)
    def test_random_i_j(self, mocked_randint):
        game = Game(
            num_rows=2,
            num_cols=2,
            up_key="w",
            down_key="s",
            left_key="a",
            right_key="d",
        )

        i, j = game.random_i_j()

        assert i == 1
        assert j == 1

        assert mocked_randint.call_count == 2
        assert mocked_randint.call_args_list[0] == mock.call(0, 1)
        assert mocked_randint.call_args_list[1] == mock.call(0, 1)

    def test_create_number(self):
        game = Game(
            num_rows=2,
            num_cols=2,
            up_key="w",
            down_key="s",
            left_key="a",
            right_key="d",
        )

        with mock.patch("random.randint", return_value=1), mock.patch(
            "random.choice", return_value=2
        ):
            game.create_number()

        assert game.mat[1, 1] == 2
        assert game.max_val == 2

    def test_create_number_when_game_is_full(self):
        game = Game(
            num_rows=2,
            num_cols=2,
            up_key="w",
            down_key="s",
            left_key="a",
            right_key="d",
        )

        game.mat[0, 0] = 2
        game.mat[0, 1] = 4
        game.mat[1, 0] = 8
        game.mat[1, 1] = 16

        old_mat = game.mat.copy()

        with mock.patch("random.randint", return_value=1) as mocked_randint, mock.patch(
            "random.choice", return_value=2
        ) as mocked_choice:
            game.create_number()

        assert mocked_randint.call_count == 0
        assert mocked_choice.call_count == 0
        assert np.array_equal(game.mat, old_mat)

    def test_create_number_with_existing_numbers(self):
        game = Game(
            num_rows=2,
            num_cols=2,
            up_key="w",
            down_key="s",
            left_key="a",
            right_key="d",
        )

        game.mat[0, 0] = 2

        with mock.patch("random.randint", side_effect=[0, 0, 1, 1]), mock.patch(
            "random.choice", return_value=4
        ):
            game.create_number()

        assert game.mat[0, 0] == 2
        assert game.mat[1, 1] == 4
        assert game.max_val == 4

    @pytest.mark.parametrize(
        argnames=["row", "expected", "move_made"],
        argvalues=[
            ([2, 0, 0, 0, 0], [2, 0, 0, 0, 0], False),
            ([0, 2, 0, 0, 0], [2, 0, 0, 0, 0], True),
            ([0, 0, 2, 0, 0], [2, 0, 0, 0, 0], True),
            ([0, 0, 0, 2, 0], [2, 0, 0, 0, 0], True),
            ([0, 0, 0, 0, 2], [2, 0, 0, 0, 0], True),
            ([2, 2, 0, 0, 0], [4, 0, 0, 0, 0], True),
            ([0, 2, 2, 0, 0], [4, 0, 0, 0, 0], True),
            ([0, 0, 2, 2, 0], [4, 0, 0, 0, 0], True),
            ([0, 0, 0, 2, 2], [4, 0, 0, 0, 0], True),
            ([2, 0, 2, 0, 0], [4, 0, 0, 0, 0], True),
            ([2, 0, 0, 2, 0], [4, 0, 0, 0, 0], True),
            ([2, 0, 0, 0, 2], [4, 0, 0, 0, 0], True),
            ([0, 2, 0, 2, 0], [4, 0, 0, 0, 0], True),
            ([0, 2, 0, 0, 2], [4, 0, 0, 0, 0], True),
            ([0, 0, 2, 0, 2], [4, 0, 0, 0, 0], True),
            ([2, 2, 2, 0, 0], [4, 2, 0, 0, 0], True),
            ([2, 2, 0, 2, 0], [4, 2, 0, 0, 0], True),
            ([2, 0, 2, 2, 0], [4, 2, 0, 0, 0], True),
            ([0, 2, 2, 2, 0], [4, 2, 0, 0, 0], True),
            ([2, 2, 2, 2, 0], [4, 4, 0, 0, 0], True),
            ([2, 2, 4, 2, 0], [4, 4, 2, 0, 0], True),
            ([2, 4, 2, 2, 0], [2, 4, 4, 0, 0], True),
            ([4, 2, 2, 2, 0], [4, 4, 2, 0, 0], True),
            ([2, 4, 4, 2, 0], [2, 8, 2, 0, 0], True),
            ([4, 2, 4, 2, 0], [4, 2, 4, 2, 0], False),
            ([4, 4, 2, 2, 0], [8, 4, 0, 0, 0], True),
            ([2, 2, 2, 4, 0], [4, 2, 4, 0, 0], True),
            ([2, 2, 0, 4, 4], [4, 8, 0, 0, 0], True),
            ([2, 0, 2, 4, 4], [4, 8, 0, 0, 0], True),
            ([0, 2, 2, 4, 4], [4, 8, 0, 0, 0], True),
            ([2, 2, 4, 4, 0], [4, 8, 0, 0, 0], True),
            ([0, 2, 4, 2, 4], [2, 4, 2, 4, 0], True),
            ([4, 2, 2, 4, 0], [4, 4, 4, 0, 0], True),
        ],
    )
    def test_move_left(self, row, expected, move_made):
        game = Game(
            num_rows=1,
            num_cols=5,
            up_key="w",
            down_key="s",
            left_key="a",
            right_key="d",
        )

        game.mat[0] = row

        assert game.move_left() == move_made
        assert np.array_equal(game.mat[0], expected)

    @pytest.mark.parametrize(
        argnames=["row", "expected", "move_made"],
        argvalues=[
            ([2, 0, 0, 0, 0], [0, 0, 0, 0, 2], True),
            ([0, 2, 0, 0, 0], [0, 0, 0, 0, 2], True),
            ([0, 0, 2, 0, 0], [0, 0, 0, 0, 2], True),
            ([0, 0, 0, 2, 0], [0, 0, 0, 0, 2], True),
            ([0, 0, 0, 0, 2], [0, 0, 0, 0, 2], False),
            ([2, 2, 0, 0, 0], [0, 0, 0, 0, 4], True),
            ([0, 2, 2, 0, 0], [0, 0, 0, 0, 4], True),
            ([0, 0, 2, 2, 0], [0, 0, 0, 0, 4], True),
            ([0, 0, 0, 2, 2], [0, 0, 0, 0, 4], True),
            ([2, 0, 2, 0, 0], [0, 0, 0, 0, 4], True),
            ([2, 0, 0, 2, 0], [0, 0, 0, 0, 4], True),
            ([2, 0, 0, 0, 2], [0, 0, 0, 0, 4], True),
            ([0, 2, 0, 2, 0], [0, 0, 0, 0, 4], True),
            ([0, 2, 0, 0, 2], [0, 0, 0, 0, 4], True),
            ([0, 0, 2, 0, 2], [0, 0, 0, 0, 4], True),
            ([2, 2, 2, 0, 0], [0, 0, 0, 2, 4], True),
            ([2, 2, 0, 2, 0], [0, 0, 0, 2, 4], True),
            ([2, 0, 2, 2, 0], [0, 0, 0, 2, 4], True),
            ([0, 2, 2, 2, 0], [0, 0, 0, 2, 4], True),
            ([2, 2, 2, 2, 0], [0, 0, 0, 4, 4], True),
            ([2, 2, 4, 2, 0], [0, 0, 4, 4, 2], True),
            ([2, 4, 2, 2, 0], [0, 0, 2, 4, 4], True),
            ([4, 2, 2, 2, 0], [0, 0, 4, 2, 4], True),
            ([2, 4, 4, 2, 0], [0, 0, 2, 8, 2], True),
            ([4, 2, 4, 2, 0], [0, 4, 2, 4, 2], True),
            ([4, 4, 2, 2, 0], [0, 0, 0, 8, 4], True),
            ([2, 2, 2, 4, 0], [0, 0, 2, 4, 4], True),
            ([2, 2, 0, 4, 4], [0, 0, 0, 4, 8], True),
            ([2, 0, 2, 4, 4], [0, 0, 0, 4, 8], True),
            ([0, 2, 2, 4, 4], [0, 0, 0, 4, 8], True),
            ([2, 2, 4, 4, 0], [0, 0, 0, 4, 8], True),
            ([0, 2, 4, 2, 4], [0, 2, 4, 2, 4], False),
            ([4, 2, 2, 4, 0], [0, 0, 4, 4, 4], True),
            ([0, 2, 4, 4, 2], [0, 0, 2, 8, 2], True),
        ],
    )
    def test_move_right(self, row, expected, move_made):
        game = Game(
            num_rows=1,
            num_cols=5,
            up_key="w",
            down_key="s",
            left_key="a",
            right_key="d",
        )

        game.mat[0] = row

        assert game.move_right() == move_made
        assert np.array_equal(game.mat[0], expected)

    @pytest.mark.parametrize(
        argnames=["col", "expected", "move_made"],
        argvalues=[
            ([2, 0, 0, 0, 0], [2, 0, 0, 0, 0], False),
            ([0, 2, 0, 0, 0], [2, 0, 0, 0, 0], True),
            ([0, 0, 2, 0, 0], [2, 0, 0, 0, 0], True),
            ([0, 0, 0, 2, 0], [2, 0, 0, 0, 0], True),
            ([0, 0, 0, 0, 2], [2, 0, 0, 0, 0], True),
            ([2, 2, 0, 0, 0], [4, 0, 0, 0, 0], True),
            ([0, 2, 2, 0, 0], [4, 0, 0, 0, 0], True),
            ([0, 0, 2, 2, 0], [4, 0, 0, 0, 0], True),
            ([0, 0, 0, 2, 2], [4, 0, 0, 0, 0], True),
            ([2, 0, 2, 0, 0], [4, 0, 0, 0, 0], True),
            ([2, 0, 0, 2, 0], [4, 0, 0, 0, 0], True),
            ([2, 0, 0, 0, 2], [4, 0, 0, 0, 0], True),
            ([0, 2, 0, 2, 0], [4, 0, 0, 0, 0], True),
            ([0, 2, 0, 0, 2], [4, 0, 0, 0, 0], True),
            ([0, 0, 2, 0, 2], [4, 0, 0, 0, 0], True),
            ([2, 2, 2, 0, 0], [4, 2, 0, 0, 0], True),
            ([2, 2, 0, 2, 0], [4, 2, 0, 0, 0], True),
            ([2, 0, 2, 2, 0], [4, 2, 0, 0, 0], True),
            ([0, 2, 2, 2, 0], [4, 2, 0, 0, 0], True),
            ([2, 2, 2, 2, 0], [4, 4, 0, 0, 0], True),
            ([2, 2, 4, 2, 0], [4, 4, 2, 0, 0], True),
            ([2, 4, 2, 2, 0], [2, 4, 4, 0, 0], True),
            ([4, 2, 2, 2, 0], [4, 4, 2, 0, 0], True),
            ([2, 4, 4, 2, 0], [2, 8, 2, 0, 0], True),
            ([4, 2, 4, 2, 0], [4, 2, 4, 2, 0], False),
            ([4, 4, 2, 2, 0], [8, 4, 0, 0, 0], True),
            ([2, 2, 2, 4, 0], [4, 2, 4, 0, 0], True),
            ([2, 2, 0, 4, 4], [4, 8, 0, 0, 0], True),
            ([2, 0, 2, 4, 4], [4, 8, 0, 0, 0], True),
            ([0, 2, 2, 4, 4], [4, 8, 0, 0, 0], True),
            ([2, 2, 4, 4, 0], [4, 8, 0, 0, 0], True),
            ([0, 2, 4, 2, 4], [2, 4, 2, 4, 0], True),
            ([4, 2, 2, 4, 0], [4, 4, 4, 0, 0], True),
        ],
    )
    def test_move_up(self, col, expected, move_made):
        game = Game(
            num_rows=5,
            num_cols=1,
            up_key="w",
            down_key="s",
            left_key="a",
            right_key="d",
        )

        game.mat[:, 0] = col

        assert game.move_up() == move_made
        assert np.array_equal(game.mat[:, 0], expected)

    @pytest.mark.parametrize(
        argnames=["col", "expected", "move_made"],
        argvalues=[
            ([2, 0, 0, 0, 0], [0, 0, 0, 0, 2], True),
            ([0, 2, 0, 0, 0], [0, 0, 0, 0, 2], True),
            ([0, 0, 2, 0, 0], [0, 0, 0, 0, 2], True),
            ([0, 0, 0, 2, 0], [0, 0, 0, 0, 2], True),
            ([0, 0, 0, 0, 2], [0, 0, 0, 0, 2], False),
            ([2, 2, 0, 0, 0], [0, 0, 0, 0, 4], True),
            ([0, 2, 2, 0, 0], [0, 0, 0, 0, 4], True),
            ([0, 0, 2, 2, 0], [0, 0, 0, 0, 4], True),
            ([0, 0, 0, 2, 2], [0, 0, 0, 0, 4], True),
            ([2, 0, 2, 0, 0], [0, 0, 0, 0, 4], True),
            ([2, 0, 0, 2, 0], [0, 0, 0, 0, 4], True),
            ([2, 0, 0, 0, 2], [0, 0, 0, 0, 4], True),
            ([0, 2, 0, 2, 0], [0, 0, 0, 0, 4], True),
            ([0, 2, 0, 0, 2], [0, 0, 0, 0, 4], True),
            ([0, 0, 2, 0, 2], [0, 0, 0, 0, 4], True),
            ([2, 2, 2, 0, 0], [0, 0, 0, 2, 4], True),
            ([2, 2, 0, 2, 0], [0, 0, 0, 2, 4], True),
            ([2, 0, 2, 2, 0], [0, 0, 0, 2, 4], True),
            ([0, 2, 2, 2, 0], [0, 0, 0, 2, 4], True),
            ([2, 2, 2, 2, 0], [0, 0, 0, 4, 4], True),
            ([2, 2, 4, 2, 0], [0, 0, 4, 4, 2], True),
            ([2, 4, 2, 2, 0], [0, 0, 2, 4, 4], True),
            ([4, 2, 2, 2, 0], [0, 0, 4, 2, 4], True),
            ([2, 4, 4, 2, 0], [0, 0, 2, 8, 2], True),
            ([4, 2, 4, 2, 0], [0, 4, 2, 4, 2], True),
            ([4, 4, 2, 2, 0], [0, 0, 0, 8, 4], True),
            ([2, 2, 2, 4, 0], [0, 0, 2, 4, 4], True),
            ([2, 2, 0, 4, 4], [0, 0, 0, 4, 8], True),
            ([2, 0, 2, 4, 4], [0, 0, 0, 4, 8], True),
            ([0, 2, 2, 4, 4], [0, 0, 0, 4, 8], True),
            ([2, 2, 4, 4, 0], [0, 0, 0, 4, 8], True),
            ([0, 2, 4, 2, 4], [0, 2, 4, 2, 4], False),
            ([4, 2, 2, 4, 0], [0, 0, 4, 4, 4], True),
            ([0, 2, 4, 4, 2], [0, 0, 2, 8, 2], True),
        ],
    )
    def test_move_down(self, col, expected, move_made):
        game = Game(
            num_rows=5,
            num_cols=1,
            up_key="w",
            down_key="s",
            left_key="a",
            right_key="d",
        )

        game.mat[:, 0] = col

        assert game.move_down() == move_made
        assert np.array_equal(game.mat[:, 0], expected)

    def test_handle_user_input_quit(self):
        game = Game(
            num_rows=4,
            num_cols=4,
            up_key="w",
            down_key="s",
            left_key="a",
            right_key="d",
        )

        mat = np.array(
            [
                [2, 2, 4, 8],
                [4, 2, 2, 4],
                [2, 4, 2, 2],
                [4, 2, 4, 2],
            ]
        )

        game.mat = mat.copy()

        with mock.patch("builtins.input", return_value="q"):
            assert game.handle_user_input() is False

        assert np.array_equal(game.mat, mat)

    def test_handle_user_input_left_key(self):
        game = Game(
            num_rows=4,
            num_cols=4,
            up_key="w",
            down_key="s",
            left_key="a",
            right_key="d",
        )

        game.mat = np.array(
            [
                [2, 2, 4, 8],
                [4, 2, 2, 4],
                [2, 4, 2, 2],
                [4, 2, 4, 2],
            ]
        )

        with mock.patch("builtins.input", return_value="a"):
            assert game.handle_user_input() is True

        # The new matrix should be:
        #   4 4 8 x
        #   4 4 4 x
        #   2 4 4 x
        #   4 2 4 2
        # where 'x' indicated a cell where a new number may be added.
        assert game.mat[0][0] == 4
        assert game.mat[0][1] == 4
        assert game.mat[0][2] == 8

        assert game.mat[1][0] == 4
        assert game.mat[1][1] == 4
        assert game.mat[1][2] == 4

        assert game.mat[2][0] == 2
        assert game.mat[2][1] == 4
        assert game.mat[2][2] == 4

        assert game.mat[3][0] == 4
        assert game.mat[3][1] == 2
        assert game.mat[3][2] == 4
        assert game.mat[3][3] == 2

        # There should only be one new number added.
        assert [game.mat[0][3], game.mat[1][3], game.mat[2][3]].count(0) == 2

    def test_handle_user_input_right_key(self):
        game = Game(
            num_rows=4,
            num_cols=4,
            up_key="w",
            down_key="s",
            left_key="a",
            right_key="d",
        )

        game.mat = np.array(
            [
                [2, 2, 4, 8],
                [4, 2, 2, 4],
                [2, 4, 2, 2],
                [4, 2, 4, 2],
            ]
        )

        with mock.patch("builtins.input", return_value="d"):
            assert game.handle_user_input() is True

        # The new matrix should be:
        #   x 4 4 8
        #   x 4 4 4
        #   x 2 4 4
        #   4 2 4 2
        # where 'x' indicated a cell where a new number may be added.
        assert game.mat[0][1] == 4
        assert game.mat[0][2] == 4
        assert game.mat[0][3] == 8

        assert game.mat[1][1] == 4
        assert game.mat[1][2] == 4
        assert game.mat[1][3] == 4

        assert game.mat[2][1] == 2
        assert game.mat[2][2] == 4
        assert game.mat[2][3] == 4

        assert game.mat[3][0] == 4
        assert game.mat[3][1] == 2
        assert game.mat[3][2] == 4
        assert game.mat[3][3] == 2

        # There should only be one new number added.
        assert [game.mat[0][0], game.mat[1][0], game.mat[2][0]].count(0) == 2

    def test_handle_user_input_up_key(self):
        game = Game(
            num_rows=4,
            num_cols=4,
            up_key="w",
            down_key="s",
            left_key="a",
            right_key="d",
        )

        game.mat = np.array(
            [
                [2, 2, 4, 8],
                [4, 2, 2, 4],
                [2, 4, 2, 2],
                [4, 2, 4, 2],
            ]
        )

        with mock.patch("builtins.input", return_value="w"):
            assert game.handle_user_input() is True

        # The new matrix should be:
        #   2 4 4 8
        #   4 4 4 4
        #   2 2 4 4
        #   4 x x x
        # where 'x' indicated a cell where a new number may be added.
        assert game.mat[0][0] == 2
        assert game.mat[0][1] == 4
        assert game.mat[0][2] == 4
        assert game.mat[0][3] == 8

        assert game.mat[1][0] == 4
        assert game.mat[1][1] == 4
        assert game.mat[1][2] == 4
        assert game.mat[1][3] == 4

        assert game.mat[2][0] == 2
        assert game.mat[2][1] == 2
        assert game.mat[2][2] == 4
        assert game.mat[2][3] == 4

        assert game.mat[3][0] == 4

        # There should only be one new number added.
        assert [game.mat[3][1], game.mat[3][2], game.mat[3][3]].count(0) == 2

    def test_handle_user_input_down_key(self):
        game = Game(
            num_rows=4,
            num_cols=4,
            up_key="w",
            down_key="s",
            left_key="a",
            right_key="d",
        )

        game.mat = np.array(
            [
                [2, 2, 4, 8],
                [4, 2, 2, 4],
                [2, 4, 2, 2],
                [4, 2, 4, 2],
            ]
        )

        with mock.patch("builtins.input", return_value="s"):
            assert game.handle_user_input() is True

        # The new matrix should be:
        #   2 x x x
        #   4 4 4 8
        #   2 4 4 4
        #   4 2 4 4
        # where 'x' indicated a cell where a new number may be added.
        assert game.mat[0][0] == 2

        assert game.mat[1][0] == 4
        assert game.mat[1][1] == 4
        assert game.mat[1][2] == 4
        assert game.mat[1][3] == 8

        assert game.mat[2][0] == 2
        assert game.mat[2][1] == 4
        assert game.mat[2][2] == 4
        assert game.mat[2][3] == 4

        assert game.mat[3][0] == 4
        assert game.mat[3][1] == 2
        assert game.mat[3][2] == 4
        assert game.mat[3][3] == 4

        # There should only be one new number added.
        assert [game.mat[0][1], game.mat[0][2], game.mat[0][3]].count(0) == 2
