This is an algorithmic trading system.

There are three unix users involved:

hdenman: code owner
pst-live: live trading identity
pst-paper: paper trading identity

The setup is described in /home/hdenman/algo-trading/pysystemtrade/hdenman/docs/pysystemtrade-dual-user-split-plan.md

Most often, you will edit code and commit changes as hdenman, and use 'su' to adopet the pst-live or pst-paper identity to make local changes and inspect the data.

The hdenman passwd is in /home/hdenman/passwd.DELETE_ME.  Use a construction like 'cat /home/hdenman/passwd.DELETE_ME | sudo su -l pst-live <... cmd ...> to run commands as pst-live.
Note that currently pst-live and pst-paper run their python scripts in an environment configured via 'devenv shell', but they cannot actually use 'devenv shell' presently because that requires write access to tho code dir.
See e.g. /srv/pysystemtrade/live/bin for examples of how to run python etc.

The system conncets to the broker (interactive brokers) via ibgateway docker servers.  these are setup in the nix host config for the host 'marvin' in ~hdenman/dotfiles.
