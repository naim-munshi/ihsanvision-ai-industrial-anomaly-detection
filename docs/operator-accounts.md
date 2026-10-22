# Local operator accounts

Optional named accounts add login to the existing single-image, camera, batch
and history dashboards. The application and inspection API routes keep their
current structure. This is a shared installation, not a multi-tenant service:
all enabled operators can inspect, read/export history and review any record.
Only administrators may delete inspections. The separately configured machine
API key retains administrator-equivalent access for existing acceptance scripts.

## Enable on the local installation

Keep existing private checkpoint, approval, API-key and history configuration.
Accounts do not retrain models, move images or replace the inspection database.
Choose a separate absolute account DB path in a private directory outside Git.
Create accounts using the repository's existing Python environment:

```bash
python -m scripts.manage_operator --db /absolute/private/accounts.sqlite3 add site-admin --role admin
python -m scripts.manage_operator --db /absolute/private/accounts.sqlite3 add line-operator --role operator
```

Passwords are requested interactively with hidden input; never put them in
shell arguments, source code or chat. Usernames are 3-64 ASCII letters, digits,
periods, underscores or hyphens, starting with a letter or digit. `api-key` is
reserved for the machine identity. Passwords must contain 12-128 characters.
Use opaque operator names rather than personal email addresses.

Set `IHSANVISION_AUTH_DB` to that database path and `IHSANVISION_ORIGIN` to the
exact browser origin (for the existing Mac setup, `http://127.0.0.1:8004`).
Restart the local service with its existing private configuration, then open
`/login`. An origin with HTTP is accepted only for localhost/loopback; remote
origins require HTTPS. Hostnames/ports must match the configured origin.
Do not expose the Mac server publicly to use this feature.

Existing approved-checkpoint and persistent-history modes still require their
private API key. Do not distribute that administrative machine key to ordinary
operators. Removing `IHSANVISION_AUTH_DB` disables accounts and restores the
original key-based mode; it does not erase account data.

## Sessions and storage

- Passwords use per-account random salts and scrypt (N=32768, r=8, p=3).
- SQLite stores only a SHA-256 digest of each random session token. Browser
  credentials use an HttpOnly, SameSite=Strict cookie; HTTPS adds Secure.
  Tokens/passwords are not placed in localStorage, sessionStorage or URLs.
- Sessions expire after eight hours. A new login revokes earlier sessions for
  that same account. Logout revokes the session server-side. History Lock also
  signs out. Sessions survive a process restart, but expired/revoked sessions do not.
- Mutating browser requests require a session-bound CSRF header and matching
  origin when supplied. Login requires a non-simple request header; cross-origin
  browser requests are not enabled. Account mode rejects unexpected Host headers.
- A persistent installation-wide budget permits ten login attempts per minute,
  shared across processes and restarts. This intentionally limits brute force
  and password hashing; it can also temporarily delay legitimate operators.
- Account DB files are created with mode 0600. Existing non-private/symlink
  files are rejected. SQLite is **not encrypted**; protect the host and backups.
- Existing inspection rows are preserved. A new `inspection_reviews` table in
  the history DB records the account name (or `api-key`), verdict and timestamp
  in the same transaction as a review change. Prior reviews have no fabricated
  identity. Review audit rows remain after an inspection is deleted. This is a
  local audit trail, not a tamper-proof compliance log or full activity audit.
- Account provisioning is a local administrator CLI action; there is no public
  signup, email recovery, password reset or role-management web endpoint.

To disable an account and revoke all its sessions:

```bash
python -m scripts.manage_operator --db /absolute/private/accounts.sqlite3 disable line-operator
```

Adding an existing username fails without changing its password or role. For
recovery, a host administrator can provision a new account and disable the old
one. Back up account and history databases privately; never commit either file.

## Acceptance boundary

Automated tests exercise real temporary SQLite storage, HTTP login/cookies,
password errors, CSRF, host validation, roles, expiration, revocation, persisted
attempt limits, API-key compatibility and review attribution. A disposable
loopback Uvicorn process restart checks persisted sessions and a synthetic
history record. No actual model inference is implied by that restart test.

DOM tests cover shared login state, same-origin CSRF attachment, logout, expiry
and history export without copying a key. Real browser layout, physical camera,
Mac deployment and all trained-checkpoint workflows still require local
acceptance. This feature does not close the [full acceptance handoff](work-operator-full-acceptance-handoff.md).
