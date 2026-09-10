import os, json, urllib.request, urllib.error
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env', override=True)
key = os.getenv('GROQ_API_KEY', '').strip()

candidates = [
    'compound-beta',
    'compound-beta-mini',
    'llama-4-scout-17b-16e-instruct',
    'llama-4-maverick-17b-128e-instruct',
    'moonshotai/kimi-k2-instruct',
    'mistral-saba-24b',
    'llama-3.1-8b-instant',
    'llama-3.3-70b-versatile',
    'llama3-8b-8192',
    'meta-llama/llama-4-scout-17b-16e-instruct',
    'meta-llama/llama-4-maverick-17b-128e-instruct',
]

for model in candidates:
    p = json.dumps({
        'model': model,
        'messages': [{'role': 'user', 'content': 'hi'}],
        'max_tokens': 5
    }).encode('utf-8')
    req = urllib.request.Request(
        'https://api.groq.com/openai/v1/chat/completions',
        data=p,
        headers={
            'Authorization': 'Bearer ' + key,
            'Content-Type': 'application/json'
        },
        method='POST'
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode('utf-8'))
            reply = data['choices'][0]['message']['content']
            print(f'[WORKS] {model} => {reply[:30]}')
    except urllib.error.HTTPError as e:
        raw = e.read().decode('utf-8', errors='replace')
        try:
            err = json.loads(raw)
            code = err.get('error', {}).get('code', '')
            msg  = err.get('error', {}).get('message', '')[:70]
            print(f'[{e.code} {code}] {model}: {msg}')
        except Exception:
            print(f'[{e.code}] {model}: {raw[:80]}')
    except Exception as ex:
        print(f'[ERR] {model}: {ex}')
