"""cli.py -- CLI for the 2048 game."""

import argparse


def _single_char(char: str) -> str:
    """Checks if a string is a single character.

    Args:
        char (str): The string to check.

    Raises:
        argparse.ArgumentTypeError: If the string is not a single character.

    Returns:
        str: The single character string.
    """
    if len(char) != 1:
        raise argparse.ArgumentTypeError(f"{char} is not a single character")

    return char


def create_parser() -> argparse.ArgumentParser:
    """Creates an ArgumentParser for the game.

    Returns:
        argparse.ArgumentParser: The ArgumentParser for the game.
    """
    parser = argparse.ArgumentParser(
        description="Play 2048 right here in your terminal!"
    )

    parser.add_argument(
        "--num-rows", type=int, default=4, help="Number of rows in the game grid"
    )
    parser.add_argument(
        "--num-cols",
        type=int,
        default=4,
        help="Number of columns in the game grid",
    )
    parser.add_argument(
        "-u",
        "--up-key",
        type=_single_char,
        default="w",
        help="The key to press to move up (default: w)",
    )
    parser.add_argument(
        "-d",
        "--down-key",
        type=_single_char,
        default="s",
        help="The key to press to move down (default: s)",
    )
    parser.add_argument(
        "-l",
        "--left-key",
        type=_single_char,
        default="a",
        help="The key to press to move left (default: a)",
    )
    parser.add_argument(
        "-r",
        "--right-key",
        type=_single_char,
        default="d",
        help="The key to press to move right (default: d)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (scroll up to view output)",
    )

    return parser
