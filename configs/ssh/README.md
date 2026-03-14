# SSH Configuration for HPC Remote-SSH

Add the following to your local `~/.ssh/config` to enable VS Code Remote-SSH
connections to UVA (Rivanna/Afton) and Frontier (OLCF) with connection
multiplexing. Multiplexing means the first connection triggers DUO/MFA, and
subsequent connections (including VS Code file browser, terminal, extensions)
reuse the same authenticated socket.

## ~/.ssh/config snippet

```sshconfig
# -----------------------------------------------------------------------
# UVA Rivanna / Afton
# -----------------------------------------------------------------------
Host uva
    HostName login1.hpc.virginia.edu
    User <your_UVA_computing_ID>
    ControlMaster auto
    ControlPath ~/.ssh/sockets/uva-%r@%h:%p
    ControlPersist 4h
    ServerAliveInterval 60
    ServerAliveCountMax 3

# -----------------------------------------------------------------------
# OLCF Frontier
# -----------------------------------------------------------------------
Host frontier
    HostName frontier.olcf.ornl.gov
    User <your_OLCF_username>
    ControlMaster auto
    ControlPath ~/.ssh/sockets/frontier-%r@%h:%p
    ControlPersist 4h
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

## Setup steps

1. Create the socket directory:
   ```bash
   mkdir -p ~/.ssh/sockets
   chmod 700 ~/.ssh/sockets
   ```

2. Paste the snippet above into `~/.ssh/config` with your usernames filled in.

3. Test the connection (this triggers DUO/MFA once):
   ```bash
   ssh uva
   ```

4. Open VS Code and use **Remote-SSH: Connect to Host** > `uva` or `frontier`.
   Subsequent connections within the `ControlPersist` window reuse the
   existing socket without re-authenticating.

## Notes

- `ControlPersist 4h` keeps the background socket alive for 4 hours after
  your last connection. Adjust to your typical work session length.
- `ServerAliveInterval 60` sends keepalive packets every 60 seconds to
  prevent idle timeout on HPC login nodes.
- Run `/setup-hpc-integration` for a guided walkthrough including Globus setup.
