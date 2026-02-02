import sys
import os

backend_dir = os.path.join(os.getcwd(), 'backend')
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from processor_service import ProcessorService

print("Initializing ProcessorService...")
processor = ProcessorService("output")

# Use the file we just scanned
# Use the workspace test file
scan_path = r"test_scanner_bed.png"

if not os.path.exists(scan_path):
    print(f"File {scan_path} not found!")
    sys.exit("Test file missing.")

print(f"Processing {scan_path}...")
try:
    results = processor.detect_and_crop(
        scan_path, 
        output_subfolder="test_process_debug",
        sensitivity=210,
        crop_margin=10,
        dpi=400,
        use_smart_detection=True
    )
    print(f"Processing SUCCESS! Found {len(results)} photos.")
    for res in results:
        print(f"  - {res['path']}")
except Exception as e:
    print(f"Processing FAILED: {e}")
    import traceback
    traceback.print_exc()
