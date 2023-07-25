"""game.py -- The main game logic for 2048."""

import os
import random

from typing import Tuple

from termcolor import cprint

import numpy as np


class Game:
    """A 2048 game."""

    STARTING_NUMBERS = (2, 4)

    def __init__(
        self,
        *,
        num_rows: int,
        num_cols: int,
        up_key: str,
        down_key: str,
        left_key: str,
        right_key: str,
        debug: bool = False,
    ) -> None:
        """Create a new Game instance.

        Args:
            num_rows (int): The number of rows in the game grid.
            num_cols (int): The number of columns in the game grid.
            debug (bool): Whether to output debug to console. Defaults to False.
        """
        self.mat = np.zeros((num_rows, num_cols), dtype=int)
        self.max_val = 0
        self.debug = debug
        self.up_key = up_key
        self.down_key = down_key
        self.left_key = left_key
        self.right_key = right_key

    def __str__(self) -> str:
        """Return a string representation of the game."""
        separator = "|" + "---------+" * (self.num_cols - 1) + "---------|"

        lines = [separator]

        for row in self.mat:
            lines.append("| " + " | ".join("       " for _ in row) + " |")
            lines.append(
                "| "
                + " | ".join(f"{self.format_number(num) : ^7}" for num in row)
                + " |"
            )
            lines.append("| " + " | ".join("       " for _ in row) + " |")
            lines.append(separator)

        lines.append("")

        return "\n".join(lines)

    def __contains__(self, num: int) -> bool:
        """Check if a number is in the game.

        Args:
            num (int): The number to check.

        Returns:
            bool: True if the number is in the game, False otherwise.
        """
        return num in self.mat

    @property
    def num_rows(self) -> int:
        """Return the number of rows in the game."""
        return self.mat.shape[0]

    @property
    def num_cols(self) -> int:
        """Return the number of columns in the game."""
        return self.mat.shape[1]

    def print_game_to_terminal(self) -> None:
        """Print the game to the terminal.

        This will clear the terminal, print the game header, and then print the game.
        """
        # Clear the terminal, handling both Windows and Unix.
        os.system("cls" if os.name == "nt" else "clear")

        self.print_header()
        print(self)

    @staticmethod
    def format_number(num: int) -> str:
        """Format a number for printing.

        Args:
            num (int): The number to format.
        """
        if num == 0:
            return " "

        return str(num)

    def print_header(self) -> str:
        """Print the game header.

        This includes the welcome message, controls, etc.
        """
        print("")
        cprint("Welcome to 2048!", "green", attrs=["bold"])
        print("")
        cprint(
            "Made by Gabriele Cirulli, brought to your terminal by Rich Barton-Cooper.",
            "green",
        )
        print("")
        cprint("Controls:", "green")
        cprint(f"  {self.left_key} -- move left", "green")
        cprint(f"  {self.right_key} -- move right", "green")
        cprint(f"  {self.down_key} -- move down", "green")
        cprint(f"  {self.up_key} -- move up", "green")
        cprint("  q -- quit", "green")
        print("")
        print("")

    def play(self) -> None:
        """Play the game.

        This is the main game loop, and will continue until the game is won or lost, or
        the user quits the game.
        """
        self.create_number()
        self.create_number()

        self.print_game_to_terminal()

        while True:
            if self.game_won():
                cprint(
                    "YOU WON! Congratulations, you excellent 2048-er!",
                    "blue",
                    attrs=["bold"],
                )
                print("")
                return

            if self.no_moves_available():
                cprint(
                    "You're out of moves! You lost. Better luck next time.",
                    "red",
                    attrs=["bold"],
                )
                print("")

                if self.max_val == 1024:
                    cprint("(And you were SO close, too!)", "red")
                    print("")
                elif self.max_val == 512:
                    cprint("(Pretty good effort though!)", "red")
                    print("")
                elif self.max_val == 256:
                    cprint("(You can do better than that!)", "red")
                    print("")
                else:
                    cprint("(You can do MUCH better than that!)", "red")
                    print("")

                return

            user_input = input("\rMove (Enter): ")

            if user_input == self.left_key:
                self.move_left()
            elif user_input == self.right_key:
                self.move_right()
            elif user_input == self.down_key:
                self.move_down()
            elif user_input == self.up_key:
                self.move_up()
            elif user_input == "q":
                print("")
                cprint("Thanks for playing!", "green", attrs=["bold"])
                print("")
                return

            self.print_game_to_terminal()

    def game_won(self) -> bool:
        """Check if the game has been won.

        Returns:
            bool: True if the game has been won, False otherwise.
        """
        if self.max_val == 2048:
            self.print_debug("Game won")
            return True

        return False

    def no_moves_available(self) -> bool:
        """Check if there are any moves available.

        Returns:
            bool: True if there are no moves available, False otherwise.
        """
        for i, j in np.ndindex(self.mat.shape):
            val = self.mat[i][j]
            if val == 0:
                # If there is an empty cell, there is a move.
                return False

            if i > 0 and val == self.mat[i - 1][j]:
                # If there is a cell to the left with the same value, there is a move.
                return False

            if j > 0 and val == self.mat[i][j - 1]:
                # If there is a cell above with the same value, there is a move.
                return False

        self.print_debug("No moves available")

        return True

    def create_number(self) -> None:
        """Create a new number in a random empty cell."""
        i, j = self.random_i_j()

        while self.contains_number(i, j):
            i, j = self.random_i_j()

        number = random.choice(self.STARTING_NUMBERS)
        self.mat[i][j] = number
        self.max_val = max(self.max_val, number)

    def random_i_j(self) -> Tuple[int, int]:
        """Return a random (i, j) tuple within the bounds of the game.

        Returns:
            Tuple[int, int]: A tuple of (i, j) coordinates.
        """
        return (
            random.randint(0, self.num_rows - 1),
            random.randint(0, self.num_cols - 1),
        )

    def move_left(self) -> None:
        """Move all numbers left, combining numbers that match.

        NB: This is the only move_* method which we need to explicitly code, since the
         other move_* methods can be achieved by combinations of matrix ops and calling
         this method. See other move_* methods for details.
        """
        move_made = False

        for i in range(self.num_rows):
            if self.move_left_in_row(i):
                move_made = True

        if move_made:
            self.create_number()

    def move_left_in_row(self, i: int) -> bool:
        """Move all numbers left in a given row, combining numbers that match.

        Args:
            i (int): The row index.

        Returns:
            bool: True if a number was moved or merged, False otherwise.
        """
        self.print_debug(f" i: {i}")

        move_made = False

        # Use a slow and fast pointer within each row to move all the row numbers left,
        # merging numbers that match.
        # The slow pointer will start at the leftmost column, and the fast pointer will
        # start at the column to the right of the slow pointer.
        j_slow = 0
        for j_fast in range(1, self.num_cols):
            j_slow, move_made_in_iteration = self.move_left_iteration(i, j_slow, j_fast)

            if move_made_in_iteration:
                move_made = True

            self.print_debug(self)

        return move_made

    def move_left_iteration(self, i: int, j_slow: int, j_fast: int) -> Tuple[int, bool]:
        """A single iteration of moving left in a given row.

        This is where the actual logic for moving left happens. This method is responsible
        for moving the slow pointer, and moving and merging cell numbers as appropriate.

        This method modifies the game matrix in place.

        Args:
            i (int): The row index.
            j_slow (int): The slow pointer column index.
            j_fast (int): The fast pointer column index.

        Returns:
            Tuple[int, bool]: A tuple of the new slow pointer column index, and whether a
             move was made in this iteration.
        """
        move_made = False

        self.print_debug(f"  j_slow: {j_slow} (val: {self.mat[i][j_slow]})")
        self.print_debug(f"  j_fast: {j_fast} (val: {self.mat[i][j_fast]})")

        if self.contains_number(i, j_fast):
            if self.mat[i][j_fast] == self.mat[i][j_slow]:
                self.print_debug("j_fast matches j_slow, adding")

                # If the fast pointer matches the slow pointer, merge the numbers (setting
                # the cell value at the fast pointer to zero), and increment the slow
                # pointer.
                #
                # Examples:
                #    s  f               sf
                #   [2, 2, 0, 0] -> [4, 0, 0, 0]
                #
                #    s     f            s  f
                #   [2, 0, 2, 0] -> [4, 0, 0, 0]
                new_val = self.mat[i][j_slow] * 2
                self.mat[i][j_slow] = new_val
                self.mat[i][j_fast] = 0
                j_slow += 1

                # Update the max value.
                self.max_val = max(self.max_val, new_val)

                move_made = True

            elif not self.contains_number(i, j_slow):
                self.print_debug(
                    "j_slow is empty, copying j_fast value to j_slow"
                )

                # If the slow pointer is empty, move the fast pointer value to the slow
                # pointer (setting the cell value at the fast pointer to zero). Do not
                # increment the slow pointer here, since we don't know if the next cell
                # in the fast pointer will be a match with the new slow pointer value.
                #
                # Examples:
                #    s  f            s  f
                #   [0, 2, 2, 0] -> [2, 0, 2, 0] (the next iteration will merge the 2s)
                self.mat[i][j_slow] = self.mat[i][j_fast]
                self.mat[i][j_fast] = 0

                move_made = True

            elif not self.contains_number(i, j_slow + 1):
                self.print_debug(
                    "j_fast does not match j_slow, copying j_fast value to empty j_slow + 1"
                )

                # If the cell to the right of the slow pointer is empty, move the fast
                # pointer value to the cell to the right of the slow pointer (setting the
                # cell value at the fast pointer to zero), and increment the slow pointer.
                #
                # Examples:
                #    s        f         s     f
                #   [2, 0, 0, 4] -> [2, 4, 0, 0]
                self.mat[i][j_slow + 1] = self.mat[i][j_fast]
                self.mat[i][j_fast] = 0
                j_slow += 1

                move_made = True

            else:
                self.print_debug(
                    "j_fast does not match j_slow, and no space at j_slow + 1, skipping j_slow"
                )

                # If the cell to the right of the slow pointer is not empty, and the fast
                # pointer does not match the slow pointer, there is nothing to change at
                # the slow pointer, so increment it.
                j_slow += 1

        else:
            # If the fast pointer is empty, do nothing and wait for the next iteration.
            # Do not increment the slow pointer here, since we don't know if the next
            # cell in the fast pointer will be a match with the current slow pointer cell.
            self.print_debug("j_fast is empty, leave j_slow unchanged")

        return j_slow, move_made

    def move_right(self) -> None:
        """Move all numbers right, combining numbers that match."""
        # Flip the game matrix horizontally, move left, then flip it back.
        self.print_debug("Flipping matrix horizontally")
        self.mat = np.fliplr(self.mat)
        self.print_debug("Moving left")
        self.move_left()
        self.print_debug("Flipping matrix back")
        self.mat = np.fliplr(self.mat)

    def move_up(self) -> None:
        """Move all numbers up, combining numbers that match."""
        # Transpose the game matrix (i <-> j), move left, then transpose it back.
        self.print_debug("Transposing matrix")
        self.mat = self.mat.transpose()
        self.print_debug("Moving left")
        self.move_left()
        self.print_debug("Transposing matrix back")
        self.mat = self.mat.transpose()

    def move_down(self) -> None:
        """Move all numbers down, combining numbers that match."""
        # Transpose the game matrix (i <-> j), move right, then transpose it back.
        self.print_debug("Transposing matrix")
        self.mat = self.mat.transpose()
        self.print_debug("Moving right")
        self.move_right()
        self.print_debug("Transposing matrix back")
        self.mat = self.mat.transpose()

    def contains_number(self, i, j) -> bool:
        """Check if the game contains a number at the given coordinates.

        Args:
            i (int): The row index.
            j (int): The column index.

        Returns:
            bool: True if the game contains a number at the given coordinates, False otherwise.
        """
        return self.mat[i][j] != 0

    def print_debug(self, *args, **kwargs):
        """Print a debug message (i.e. only print to the terminal if debug is True)."""
        if self.debug:
            print(*args, **kwargs)
