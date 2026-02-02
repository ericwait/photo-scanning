import sys
import os
import time

backend_dir = os.path.join(os.getcwd(), 'backend')
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from scanner_service import ScannerService

print("Initializing ScannerService...")
try:
    scanner = ScannerService("scans")
    print("Initialized.")
except Exception as e:
    print(f"Init Failed: {e}")
    sys.exit(1)

timestamp = int(time.time())
filename = f"test_scan_{timestamp}.tiff"

print(f"Starting Scan for {filename}...")
try:
    # We call scan_page directly
    # This matches main.py logic
    path = scanner.scan_page(filename, dpi=150) # Use low DPI for speed
    print(f"Scan SUCCESS! Path: {path}")
except Exception as e:
    print(f"Scan FAILED: {e}")
    import traceback
    traceback.print_exc()
