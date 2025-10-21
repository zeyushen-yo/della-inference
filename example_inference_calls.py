"""
Minimal examples for vLLM (OpenAI-compatible) and SGLang (/generate).
"""


def vllm_openai(api_url: str, model_name: str):
    from openai import OpenAI
    client = OpenAI(base_url=f"{api_url.rstrip('/')}/v1", api_key="token-abc123")
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Respond friendly to the user."},
            {"role": "user", "content": "Hello World!"},
        ],
        max_tokens=64,
        temperature=0,
    )
    print(completion.choices[0].message)


def sglang_generate(base_url: str, model_name: str):
    import urllib.request, json
    url = f"{base_url.rstrip('/')}/generate"
    req = urllib.request.Request(url, method="POST")
    req.add_header("Content-Type", "application/json")
    payload = {
        "model": model_name,
        "text": "In one sentence, say hello from SGLang.",
        "sampling_params": {"max_new_tokens": 64, "temperature": 0},
    }
    with urllib.request.urlopen(req, data=json.dumps(payload).encode("utf-8")) as r:
        resp = json.loads(r.read().decode("utf-8"))
    print(resp.get("text", resp))


if __name__ == "__main__":
    import os, sys
    backend = os.environ.get("BACKEND", "sglang").lower()
    if backend == "sglang":
        base_url = os.environ.get("SGLANG_URL", "http://localhost:30000")
        model = os.environ.get("MODEL_PATH")
        if not model:
            print("Set MODEL_PATH to your local model path (e.g., /path/to/model)", file=sys.stderr)
            sys.exit(1)
        sglang_generate(base_url, model)
    elif backend == "vllm":
        api_url = os.environ.get("VLLM_URL", "http://localhost:8000")
        model = os.environ.get("VLLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
        vllm_openai(api_url, model)
    else:
        print("Unknown BACKEND (expected 'sglang' or 'vllm')", file=sys.stderr)
        sys.exit(2)
