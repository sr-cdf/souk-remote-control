# souk-remote-control

## Remote control readout for the SO:UK Firmware

Adapted from `souk_poll_acc.py` in `https://github.com/realtimeradio/souk-firmware/tree/main/software/control_sw/test_scripts`

Assumes `souk-firmware` has been setup and is able to run on the board.


### Server/Client model:

The server runs on the RFSoC ARM. It listens for commands on the network and runs the corresponding scripts using `souk-firmware`.

The client runs elsewhere. It sends commands to the server, receives data and performs basic analysis. It can also remotely start/stop the server.


### Setup:

A copy of `souk-remote-control-server` is needed on the RFSoC.
A copy of `souk-remote-control-client` is needed on a laptop/desktop/whatever that can see the RFSoC on the network.

It is plausible that the client could be executed directly on the RFSoC.

1. Clone this git repo to your laptop/desktop.

2. Copy the directory named `souk-remote-control-server` to '/home/casper/src' on the RFSoC, for example: `cd souk-remote-control && scp -r souk-remote-control-server casper@<rfsoc-host>:src/` (Alternatively, if the board can see the outside world, ssh in and clone the repo directly into the board, under `/home/casper/src/`.)

3. On the laptop/desktop, run the scripts in `souk-remote-control-client`



## Tested Firmware Version

This software has been tested with the SO:UK Firmware version 6.3.0.0. Please ensure that your firmware version matches this to guarantee compatibility. If you are using a different firmware version, please be aware that this software may not work as expected.