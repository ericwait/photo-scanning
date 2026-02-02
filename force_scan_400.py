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
filename = f"test_scan_400_{timestamp}.tiff"

print(f"Starting Scan (400 DPI) for {filename}...")
try:
    path = scanner.scan_page(filename, dpi=400)
    print(f"Scan SUCCESS! Path: {path}")
except Exception as e:
    print(f"Scan FAILED: {e}")
    import traceback
    traceback.print_exc()
