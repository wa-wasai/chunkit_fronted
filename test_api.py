import requests
import json

def test_stream_api():
    url = "http://localhost:8000/query"
    data = {"query": "什么是dp？"}
    
    response = requests.post(
        url, 
        json=data, 
        stream=True,
        headers={'Content-Type': 'application/json'}
    )
    
    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith('data: '):
                data_str = line_str[6:]
                if data_str.strip():
                    try:
                        chunk = json.loads(data_str)
                        if 'delta' in chunk:
                            print(chunk['delta'], end='', flush=True)
                        if chunk.get('finished'):
                            break
                    except json.JSONDecodeError:
                        continue

if __name__ == "__main__":
    test_stream_api()