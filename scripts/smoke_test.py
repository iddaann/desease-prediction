"""Production smoke test for a running HealthPredict instance.

Usage:
    python scripts/smoke_test.py https://your-service.onrender.com
"""
import json
import sys
import urllib.error
import urllib.request

def request(url, method="GET", payload=None):
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=20) as response:
        body = response.read().decode()
        return response.status, json.loads(body) if body else {}

def main():
    base = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
    checks = []

    status, health = request(base + "/health")
    checks.append(("health", status == 200 and health.get("status") in {"ok", "degraded"}))
    print("health:", status, health)

    status, symptoms = request(base + "/api/symptoms")
    checks.append(("symptoms", status == 200 and bool(symptoms.get("symptoms"))))
    print("symptoms:", status, len(symptoms.get("symptoms", [])), "features")

    status, result = request(
        base + "/api/chat",
        "POST",
        {"message": "Sejak kemarin saya demam, batuk dan kepala terasa sakit."},
    )
    checks.append(("chat", status == 200 and bool(result.get("response_text") or result.get("message"))))
    print("chat:", status, result.get("status"), result.get("disease"), result.get("confidence_score"))

    failed = [name for name, ok in checks if not ok]
    if failed:
        print("SMOKE TEST FAILED:", ", ".join(failed))
        raise SystemExit(1)

    print("SMOKE TEST PASSED")

if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as exc:
        print("HTTP ERROR:", exc.code, exc.read().decode(errors="replace"))
        raise SystemExit(1)
    except Exception as exc:
        print("REQUEST ERROR:", exc)
        raise SystemExit(1)
