# Security

Keep API keys in your local ignored `.env` or environment. Never send keys in issues,
logs, screenshots, URLs, command arguments, or public reports. Rotate any exposed key.
Use one active paid runner and preserve its request ledger, including uncertain attempts.

Report vulnerabilities through GitHub private vulnerability reporting when enabled.
Do not post exploit details or credentials in public issues. Until private reporting is
enabled, contact the repository owner privately without including live credentials.

The local web interface binds to loopback by default. Do not expose it publicly.
Paid provider data and derived research are excluded from public source releases.
MIT covers this project's code, not data redistribution or third-party dependencies.

Before release, scan the complete exported Git history with Gitleaks, audit locked
Python dependencies, run the offline tests, and inspect package contents. A clean
scanner result does not prove that research facts or personal information are absent.
