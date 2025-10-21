import json
import os
import sys
import time
import traceback
import urllib.request
import urllib.error


def fetch_model_info(base_url: str):
    url = f"{base_url.rstrip('/')}/get_model_info"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read().decode("utf-8")
    try:
        return json.loads(raw)
    except Exception:
        return {"raw": raw}


def try_generate(base_url: str, payload: dict):
    url = f"{base_url.rstrip('/')}/generate"
    req = urllib.request.Request(url, method="POST")
    req.add_header("Content-Type", "application/json")
    t0 = time.time()
    with urllib.request.urlopen(req, data=json.dumps(payload).encode("utf-8"), timeout=300) as r:
        raw = r.read().decode("utf-8")
    dt = time.time() - t0
    try:
        resp = json.loads(raw)
    except Exception:
        resp = {"raw": raw}
    return dt, resp


def main():
    port = int(os.environ.get("SGLANG_PORT", "30000"))
    base_url = f"http://localhost:{port}"
    info = fetch_model_info(base_url)
    print("model_info=", info)

    model_path = os.environ.get("MODEL_PATH", info.get("served_model_name"))

    # Primary attempt: include model path
    payload = {
        "model": model_path,
        "text": "In one sentence, say hello from SGLang.",
        "sampling_params": {"max_new_tokens": 64, "temperature": 0},
    }
    try:
        dt, resp = try_generate(base_url, payload)
        print(f"ok_with_model latency_sec={dt:.3f}")
        print(resp.get("text", resp))
        return
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print("HTTPError with model:", e.code, body, file=sys.stderr)
    except Exception:
        print("Exception with model:")
        traceback.print_exc()

    # Fallback: omit model field
    payload2 = {
        "text": "In one sentence, say hello from SGLang (no model field).",
        "sampling_params": {"max_new_tokens": 64, "temperature": 0},
    }
    try:
        dt, resp = try_generate(base_url, payload2)
        print(f"ok_without_model latency_sec={dt:.3f}")
        print(resp.get("text", resp))
        return
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print("HTTPError without model:", e.code, body, file=sys.stderr)
        sys.exit(1)
    except Exception:
        print("Exception without model:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
