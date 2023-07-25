"""args.py -- Hold the arguments for the game."""

import argparse

from typing import Optional


class Args:
    """A class to hold the arguments for the game."""

    DEFAULT_NUM_ROWS = 4
    DEFAULT_NUM_COLS = 4
    DEFAULT_UP_KEY = "w"
    DEFAULT_DOWN_KEY = "s"
    DEFAULT_LEFT_KEY = "a"
    DEFAULT_RIGHT_KEY = "d"

    def __init__(
        self,
        *,
        num_rows: Optional[int] = None,
        num_cols: Optional[int] = None,
        debug: bool = False,
        up_key: Optional[str] = None,
        down_key: Optional[str] = None,
        left_key: Optional[str] = None,
        right_key: Optional[str] = None,
    ) -> None:
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.debug = debug
        self.up_key = up_key
        self.down_key = down_key
        self.left_key = left_key
        self.right_key = right_key

    @classmethod
    def from_cli(cls, cli: argparse.Namespace) -> "Args":
        """Creates an Args object from a CLI Namespace.

        Args:
            cli (argparse.Namespace): The CLI Namespace to create an Args object from.

        Returns:
            Args: The Args object created from the CLI Namespace.
        """
        return cls(
            num_rows=cli.num_rows or cls.DEFAULT_NUM_ROWS,
            num_cols=cli.num_cols or cls.DEFAULT_NUM_COLS,
            debug=cli.debug,
            up_key=cli.up_key or cls.DEFAULT_UP_KEY,
            down_key=cli.down_key or cls.DEFAULT_DOWN_KEY,
            left_key=cli.left_key or cls.DEFAULT_LEFT_KEY,
            right_key=cli.right_key or cls.DEFAULT_RIGHT_KEY,
        )
