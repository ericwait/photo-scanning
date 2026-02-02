import sys
import os

backend_dir = os.path.join(os.getcwd(), 'backend')
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from processor_service import ProcessorService

print("Initializing ProcessorService...")
processor = ProcessorService("output")

# Use the file we just scanned
scan_path = r"scans\test_scan_1769967800.tiff"

if not os.path.exists(scan_path):
    # Fallback to test_image.jpg if the specific timestamp one is gone/renamed (unlikely)
    # Actually I need to know the filename. I'll list the directory in Python to find it if needed.
    # But for now hardcode what I saw in the log.
    print(f"File {scan_path} not found!")
    # Try to find any tiff
    files = [f for f in os.listdir("scans") if f.endswith(".tiff")]
    if files:
        scan_path = os.path.join("scans", files[-1])
        print(f"Using found file: {scan_path}")
    else:
        sys.exit("No scan file found.")

print(f"Processing {scan_path}...")
try:
    results = processor.detect_and_crop(
        scan_path, 
        output_subfolder="test_process_debug",
        sensitivity=210,
        crop_margin=10
    )
    print(f"Processing SUCCESS! Found {len(results)} photos.")
    for res in results:
        print(f"  - {res['path']}")
except Exception as e:
    print(f"Processing FAILED: {e}")
    import traceback
    traceback.print_exc()
