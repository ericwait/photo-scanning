import urllib.request
import json
import time

url = "http://localhost:8001/scan"
payload = {
    "mock": True,
    "mock_source": "test_image.jpg",
    "album_name": "debug_test",
    "sensitivity": 210,
    "crop_margin": 10
}

data = json.dumps(payload).encode('utf-8')
req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})

print(f"Triggering MOCK scan at {url}...")
try:
    with urllib.request.urlopen(req) as response:
        print(f"Status: {response.status}")
        result = json.loads(response.read().decode('utf-8'))
        print("Success!")
        # print(json.dumps(result, indent=2))
except Exception as e:
    print(f"Mock Scan Failed: {e}")
    try:
        if hasattr(e, 'read'):
            print(e.read().decode('utf-8'))
    except:
        pass
