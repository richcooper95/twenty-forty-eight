"""run.py -- Runs the game."""

from .args import Args
from .cli import create_parser
from .game import Game


def run():
    """Run the game."""
    parser = create_parser()
    cli = parser.parse_args()

    args = Args.from_cli(cli)

    game = Game(
        num_rows=args.num_rows,
        num_cols=args.num_cols,
        debug=args.debug,
        up_key=args.up_key,
        down_key=args.down_key,
        left_key=args.left_key,
        right_key=args.right_key,
    )

    game.play()
