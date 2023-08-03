# `twenty_forty_eight`
## 2️⃣0️⃣4️⃣8️⃣ in your terminal!

A way to play the famous [2048](https://play2048.co/) game from right within your terminal.

I don't own the idea behind this game; that was [Gabriele Cirulli](http://gabrielecirulli.com/). This is just a reworking.

## Installation

There are two steps to installing this command:
  1. Clone this repo on your machine.
  2. Run the `./bin/build` script and follow the instructions.

NB: The executable is very slow when loading, but fast to play. I'll probably come back to speed up loading in the future.

### Details

For ease of distribution, the `./bin/build` script will use [PyInstaller](https://pyinstaller.org/en/stable/) to generate an executable specific to your machine.

The steps performed in `./bin/build` are fairly simple - it will:
  1. Create a disposable virtual env.
  2. Activate the virtual env and install dependencies.
  3. Run `pyinstaller` inside the virtual env to create the executable.
  4. Clean up.

## Development

### Setup

To contribute, you can get set up by:
  1. Cloning this repo.
  2. Creating a virtual environment (e.g. `python3 -m venv .venv`).
  3. Installing the dev dependencies: `pip install -r requirements_dev.txt`.

### Tests

TBC.
