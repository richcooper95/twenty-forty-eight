# twenty_forty_eight -- 2048 in your terminal!

A way to play the famous [2048](https://play2048.co/) game on your terminal (so you can do it at work with no one noticing).

I don't own the idea behind this game; that was [Gabriele Cirulli](http://gabrielecirulli.com/). This is just a reworking for your terminal.

## Installation

To install this command, clone this repo and run the `./bin/build` script. Follow the instructions there to access the
`twenty_forty_eight` binary from anywhere.

### Details

For ease of distribution, the `./bin/build` script will use [PyInstaller](https://pyinstaller.org/en/stable/) to generate an executable specific to your machine.

The build steps are fairly simple:
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
