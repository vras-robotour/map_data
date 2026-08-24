# Security Policy

## Supported versions

Security fixes land on `master` and ship in the next release. Only the latest release is
supported — there are no maintained backport branches.

| Version | Supported |
|---------|-----------|
| 1.3.x   | ✅ |
| < 1.3   | ❌ |

## Reporting a vulnerability

**Do not open a public issue for a security vulnerability.**

Report it privately via one of:

- GitHub's [private vulnerability reporting](https://github.com/vras-robotour/map_data/security/advisories/new)
  (Security → Report a vulnerability)
- Email: <vlkjan6@fel.cvut.cz>

Please include the affected version or commit, what an attacker can achieve, and the steps
or a minimal `.mapdata`/`.gpx` file needed to reproduce it. You can expect an initial
response within two weeks. This is an academic research project maintained part-time, so
please allow reasonable time for a fix before any public disclosure.

## Threat model

The interactive viewer (`map_data_viewer`) is the main security-relevant surface. It is
designed for a trusted network — a workstation or an isolated robot LAN — and binds to
`127.0.0.1` by default.

An unauthenticated client that can reach the viewer can create and overwrite files in the
data directory, spawn a `wormhole send` subprocess, and trigger outbound Overpass API
queries. Two opt-in environment variables (`MAP_DATA_ACCESS_TOKEN` and
`MAP_DATA_CORS_ORIGINS`) harden that surface; the full threat model, including the CSRF
design and the cookie/header rules, is documented in the
[viewer deployment security section](https://vras-robotour.github.io/map_data/viewer/#deployment-security).

Reports we consider in scope:

- Bypassing the access-token gate or the CSRF protections
- Path traversal or arbitrary file write outside the configured data directory
- Stored or reflected XSS in the viewer frontend
- Command injection via the wormhole subprocess or any user-supplied path
- Resource-exhaustion vectors that survive the documented request and grid-size caps

Out of scope: anything that requires binding the viewer to an untrusted network without
setting `MAP_DATA_ACCESS_TOKEN`, since that configuration is documented as unauthenticated
by design.
