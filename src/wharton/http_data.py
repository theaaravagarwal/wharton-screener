"""Read-only HTTP helper that never includes request URLs/credentials in errors."""

import time

import requests


def get_json(session, url, params, provider, *, expected_type=dict):
    for attempt in range(3):
        try:
            response = session.get(url, params=params, timeout=(5, 30), allow_redirects=False)
        except requests.RequestException:
            if attempt < 2:
                time.sleep(2**attempt)
                continue
            raise ValueError(f"{provider} connection failed after 3 attempts") from None
        status = response.status_code
        if status == 200:
            try:
                payload = response.json()
            except ValueError:
                raise ValueError(f"{provider} returned invalid JSON") from None
            if not isinstance(payload, expected_type):
                raise ValueError(f"{provider} returned an unexpected response")
            return payload
        if (status == 429 or 500 <= status < 600) and attempt < 2:
            time.sleep(2**attempt)
            continue
        reason = {
            401: "check credentials",
            403: "check credentials and data entitlement",
            429: "rate limit reached",
        }.get(status, "request rejected")
        raise ValueError(f"{provider} HTTP {status}: {reason}")
    raise ValueError(f"{provider} retry limit reached")
