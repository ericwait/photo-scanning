import sys
import os

# Ensure backend directory is in path
backend_dir = os.path.join(os.getcwd(), 'backend')
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

print(f"Checking TWAIN availability...")
try:
    import twain
    print(f"TWAIN module found: {twain.__file__}")
except ImportError:
    print("TWAIN module NOT found.")

print("\nAttempting to initialize ScannerService...")
try:
    from scanner_service import ScannerService
    scanner = ScannerService("scans")
    print("ScannerService initialized.")
except Exception as e:
    print(f"Failed to initialize ScannerService: {e}")
    sys.exit(1)

print("\n--- Diagnostic: TWAIN Source Listing ---")
try:
    # 1. Initialize Source Manager
    sm = twain.SourceManager(0)
    print("Source Manager initialized.")
    
    # 2. List Sources (if possible via wrapper logic or manual attempt)
    # The pytwain wrapper doesn't have a direct 'GetDataSources' sometimes depending on version? 
    # Actually, in the code we saw 'GetDataSources' was commented out as invalid.
    # We can try to OpenSource() with a dialog if we weren't headless, but we are.
    
    # Let's try to just open the default source
    print("Attempting to open default source...")
    try:
        ss = sm.OpenSource()
        if ss:
            print(f"SUCCESS: Connected to default source: {ss.GetIdentity()['ProductName']}")
            ss.destroy()
        else:
            print("FAILED: OpenSource() returned None (User cancelled or no default).")
    except Exception as e:
        print(f"FAILED: OpenSource() raised exception: {e}")

    # Try specific Epson
    target = "EPSON Perfection V600"
    print(f"Attempting to open specific source: {target}")
    try:
        ss = sm.OpenSource(target)
        if ss:
            print(f"SUCCESS: Connected to {target}")
            ss.destroy()
        else:
            print(f"FAILED: Could not open {target}")
    except Exception as e:
        print(f"FAILED: Exception opening {target}: {e}")

    sm.destroy()

except Exception as e:
    print(f"TWAIN Diagnostic Error: {e}")
    
print("\n--- Diagnostic: WIA Device Listing ---")
try:
    import win32com.client
    device_manager = win32com.client.Dispatch("WIA.DeviceManager")
    count = device_manager.DeviceInfos.Count
    print(f"Found {count} WIA devices.")
    for i in range(1, count + 1):
        info = device_manager.DeviceInfos(i)
        print(f"  Device {i}: {info.Properties['Name'].Value} (Type: {info.Type})")
except Exception as e:
    print(f"WIA Diagnostic Error: {e}")
