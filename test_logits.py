import httpx

client = httpx.Client(timeout=30)

# 1. logprobs in /v1/completions
resp = client.post('http://127.0.0.1:8090/v1/completions', json={
    'prompt': 'Hello',
    'max_tokens': 3,
    'temperature': 0.0,
    'logprobs': 1,
})
print("1. /v1/completions + logprobs:")
data = resp.json()
if 'choices' in data:
    choice = data['choices'][0]
    print(f"   text: {repr(choice.get('text', ''))}")
    print(f"   logprobs: {choice.get('logprobs', 'NONE')}")
else:
    print(f"   {data}")

# 2. logprobs in /v1/chat/completions
resp = client.post('http://127.0.0.1:8090/v1/chat/completions', json={
    'messages': [{'role': 'user', 'content': 'Hi'}],
    'max_tokens': 3,
    'temperature': 0.0,
    'logprobs': True,
    'top_logprobs': 5,
})
print("\n2. /v1/chat/completions + logprobs:")
data = resp.json()
if 'choices' in data:
    choice = data['choices'][0]
    lp = choice.get('logprobs', None)
    print(f"   text: {repr(choice.get('message', {}).get('content', ''))}")
    if lp:
        tokens = lp.get('tokens', [])
        top = lp.get('top_logprobs', [])
        print(f"   tokens: {tokens}")
        for t, tops in zip(tokens, top):
            print(f"   '{t}' top_logprobs: {dict(list(tops.items())[:5])}")
    else:
        print(f"   logprobs: NONE")
else:
    print(f"   {data}")

# 3. Check embeddings
resp = client.post('http://127.0.0.1:8090/embeddings', json={'content': 'Hello'})
emb = resp.json()
if isinstance(emb, list):
    vec = emb[0]['embedding']
    print(f"\n3. /embeddings: dim={len(vec)}, sample={vec[:3]}")
else:
    print(f"\n3. /embeddings: {emb}")

# 4. llama-cpp-python
try:
    import llama_cpp
    print(f"\n4. llama-cpp-python: {llama_cpp.__version__}")
except ImportError:
    print("\n4. llama-cpp-python: NOT INSTALLED")

# 5. Check what endpoints are available
for path in ['/health', '/slots', '/metrics', '/props', '/v1/models']:
    try:
        resp = client.get(f'http://127.0.0.1:8090{path}')
        print(f"\n5. GET {path}: {resp.status_code} - {resp.text[:100]}")
    except Exception as e:
        print(f"\n5. GET {path}: {e}")
