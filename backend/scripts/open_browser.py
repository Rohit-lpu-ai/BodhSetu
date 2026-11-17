"""Open the default browser once the NiyamSetu server is listening."""

from __future__ import annotations

import sys
import time
import urllib.request
import webbrowser


def main() -> None:
    port = sys.argv[1] if len(sys.argv) > 1 else "8000"
    path = sys.argv[2] if len(sys.argv) > 2 else ""
    if path.startswith("/"):
        path = path.lstrip("/")

    target_url = f"http://127.0.0.1:{port}/{path}"
    health_url = f"http://127.0.0.1:{port}/api/health"

    # Wait up to 20 seconds for the server to be ready
    for _ in range(40):
        try:
            req = urllib.request.Request(
                health_url,
                headers={"User-Agent": "NiyamSetu-Startup-Check/1.0"},
            )
            with urllib.request.urlopen(req, timeout=1) as resp:
                if resp.status == 200:
                    break
        except Exception:
            time.sleep(0.5)

    # Small breathing room for static mount initialization
    time.sleep(0.5)
    webbrowser.open(target_url)


if __name__ == "__main__":
    main()
