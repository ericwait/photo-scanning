import urllib.request
print("Starting...")
try:
    with urllib.request.urlopen("http://localhost:8001/health") as response:
        print(f"Health: {response.read().decode('utf-8')}")
except Exception as e:
    print(f"Health check failed: {e}")
print("Finished.")
