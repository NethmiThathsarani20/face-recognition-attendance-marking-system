#!/usr/bin/env python3
"""
find_ip_by_mac.py

Find the IPv4 address(es) for a given MAC address on the local network.

Strategy:
- Determine the local IPv4 address
- Assume a prefix (default /24) or use provided prefix
- Ping all hosts in the subnet (parallel) to populate the ARP table
- Run `arp -a` and parse entries, normalizing MAC formats

Works on Windows, macOS and Linux (uses platform-specific ping flags).

Usage:
    python find_ip_by_mac.py 80:f3:da:62:14:c0

Optional flags: --prefix, --timeout, --workers, --no-ping
"""
from __future__ import annotations

import argparse
import ipaddress
import platform
import re
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional


def normalize_mac(mac: str) -> str:
    """Normalize MAC to lower-case colon-separated format."""
    s = mac.strip().lower()
    s = s.replace('-', ':')
    s = s.replace('.', '')
    # some devices show aabb.ccdd.eeff style; try to insert colons
    if re.fullmatch(r'[0-9a-f]{12}', s):
        s = ':'.join(s[i:i+2] for i in range(0, 12, 2))
    # if there are groups without colons but with length 12, handled above
    return s


def get_default_ip() -> str:
    """Get a non-loopback local IPv4 address by creating a UDP socket.

    This doesn't send any packets to the remote host.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't need to be reachable; this is only to pick the right interface
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        # fallback
        ip = socket.gethostbyname(socket.gethostname())
    finally:
        s.close()
    return ip


def ping_host(ip: str, timeout_ms: int = 200) -> bool:
    """Ping a host once. Return True if the ping command succeeded (exit code 0).

    timeout_ms is used differently per-platform. Keep output quiet.
    """
    system = platform.system()
    if system == 'Windows':
        # -n 1 : send 1 echo request
        # -w timeout in milliseconds
        cmd = ['ping', '-n', '1', '-w', str(timeout_ms), ip]
    elif system == 'Darwin':
        # macOS ping uses -c for count and -W for timeout in milliseconds per packet
        # macOS BSD ping's -W is in milliseconds on newer macOS, but can behave differently,
        # so keep timeout small and rely on overall performance.
        cmd = ['ping', '-c', '1', '-W', str(int(timeout_ms)) , ip]
    else:
        # Linux: -c 1 count, -W timeout in seconds (integer)
        to_sec = max(1, int((timeout_ms + 999) // 1000))
        cmd = ['ping', '-c', '1', '-W', str(to_sec), ip]

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def populate_arp_table(network: ipaddress.IPv4Network, timeout_ms: int = 200, workers: int = 100) -> None:
    """Ping all hosts in the network (concurrently) to populate ARP table."""
    ips = [str(ip) for ip in network.hosts()]
    # Limit workers to a reasonable number
    workers = min(workers, len(ips))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(ping_host, ip, timeout_ms): ip for ip in ips}
        # iterate to keep progress and allow cancellation if needed
        for fut in as_completed(futures):
            _ip = futures[fut]
            try:
                _ = fut.result()
            except Exception:
                pass


def parse_arp_table() -> List[tuple[str, str]]:
    """Run `arp -a` and parse the output. Return list of (ip, mac) pairs.

    The output format differs by OS; use a permissive regex.
    """
    try:
        out = subprocess.check_output(['arp', '-a'], text=True, stderr=subprocess.DEVNULL)
    except Exception as e:
        # On some systems `arp` might not be available
        raise RuntimeError(f"Failed to run arp: {e}")

    entries: List[tuple[str, str]] = []
    # Match lines containing IPv4 address and MAC-like token (hyphens or colons or plain hex)
    # Example windows line:  192.168.1.100           80-f3-da-62-14-c0     dynamic
    arp_line_re = re.compile(r"(\d+\.\d+\.\d+\.\d+)\s+([0-9a-fA-F:\-]{11,17})")

    for line in out.splitlines():
        m = arp_line_re.search(line)
        if m:
            ip = m.group(1)
            mac = m.group(2).lower()
            mac = mac.replace('-', ':')
            entries.append((ip, mac))

    return entries


def find_ips_by_mac(target_mac: str, entries: List[tuple[str, str]]) -> List[str]:
    target = normalize_mac(target_mac)
    found: List[str] = []
    for ip, mac in entries:
        # normalize parsed mac
        norm = normalize_mac(mac)
        if norm == target:
            found.append(ip)
    return found


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description='Find IP(s) on local network for a given MAC address.')
    p.add_argument('mac', help='MAC address to look for (e.g. 80:f3:da:62:14:c0)')
    p.add_argument('--prefix', '-p', type=int, default=24, help='CIDR prefix length (default: 24)')
    p.add_argument('--timeout', '-t', type=int, default=200, help='Ping timeout in milliseconds (default: 200)')
    p.add_argument('--workers', '-w', type=int, default=100, help='Number of parallel pings (default: 100)')
    p.add_argument('--no-ping', action='store_true', help='Do not ping hosts; just parse current ARP table')
    p.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = p.parse_args(argv)

    mac = args.mac
    if args.verbose:
        print(f"Target MAC: {mac}")

    try:
        local_ip = get_default_ip()
    except Exception:
        print("Failed to determine local IP address.")
        return 2

    if args.verbose:
        print(f"Local IP discovered: {local_ip}")

    try:
        network = ipaddress.ip_network(f"{local_ip}/{args.prefix}", strict=False)
    except Exception as e:
        print(f"Invalid network/prefix: {e}")
        return 2

    if args.verbose:
        print(f"Scanning network: {network.with_prefixlen}")

    if not args.no_ping:
        try:
            populate_arp_table(network, timeout_ms=args.timeout, workers=args.workers)
        except KeyboardInterrupt:
            print("Scan cancelled by user.")
            return 130

    try:
        entries = parse_arp_table()
    except RuntimeError as e:
        print(e)
        return 2

    if args.verbose:
        print(f"ARP entries parsed: {len(entries)}")

    matches = find_ips_by_mac(mac, entries)
    if matches:
        print("Found IP(s) for MAC", mac)
        for ip in matches:
            print(ip)
        return 0
    else:
        print(f"No IP found for MAC {mac} in ARP table.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
