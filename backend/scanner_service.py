import os
import subprocess
from typing import Optional

class ScannerService:
    def __init__(self, scan_dir: str):
        self.scan_dir = scan_dir
        if not os.path.exists(scan_dir):
            os.makedirs(scan_dir)

    def scan_page(self, filename: str) -> str:
        """
        Triggers a scan from the default scanner using Windows Image Acquisition (WIA) automation.
        Bypasses the "Select Device" and "Scan Properties" UI prompts by connecting directly.
        """
        output_path = os.path.join(self.scan_dir, filename)
        
        try:
            import win32com.client
            
            # 1. Access Device Manager to find scanners
            device_manager = win32com.client.Dispatch("WIA.DeviceManager")
            
            detector = None
            # Iterate (1-indexed) through devices to find a scanner (Type = 1)
            for i in range(1, device_manager.DeviceInfos.Count + 1):
                info = device_manager.DeviceInfos(i)
                if info.Type == 1: # Scanner
                    detector = info
                    break
            
            if not detector:
                raise Exception("No scanner device found. Please ensure it is connected and turned on.")
                
            print(f"Connecting to scanner: {detector.Properties['Name'].Value}...")
            # 2. Connect to the device
            device = detector.Connect()
            
            # 3. Get the scanner item (the flatbed) - usually Item(1)
            item = device.Items(1)
            
            # --- Configure Scan Settings ---
            # 6146: Current Intent (1=Color, 2=Grayscale, 4=Text)
            # 6147: Horizontal Resolution (DPI)
            # 6148: Vertical Resolution (DPI)
            
            try:
                # Force Color Intent
                for prop in item.Properties:
                    if prop.PropertyID == 6146: # Current Intent
                        prop.Value = 1 # Color
                        break
                
                # Set DPI to 400
                for prop in item.Properties:
                    if prop.PropertyID == 6147: # Horizontal Res
                        prop.Value = 400
                    elif prop.PropertyID == 6148: # Vertical Res
                        prop.Value = 400
                        
                print("Scanner configured: Color, 400 DPI")
            except Exception as e:
                print(f"Warning: Could not set scanner properties: {e}")

            # 4. Transfer the image
            print("Scanning in progress (UI suppressed)...")
            # FormatID for BMP: {B96B3CAB-0728-11D3-9D7B-0000F81EF32E}
            image_file = item.Transfer("{B96B3CAB-0728-11D3-9D7B-0000F81EF32E}")
            
            if image_file:
                if os.path.exists(output_path):
                    os.remove(output_path)
                    
                image_file.SaveFile(output_path)
                print(f"Scan saved to {output_path}")
                return output_path
            else:
                raise Exception("No image data returned from scanner.")
                
        except Exception as e:
            # Check for cancellation
            if "0x80210064" in str(e): # WIA_ERROR_USER_CANCELLED
                raise Exception("Scan cancelled by user.")
            
            print(f"WIA Scan Error: {e}")
            # Fallback suggestion
            if "win32com" in str(e) or "ModuleNotFoundError" in str(e):
                raise Exception("pywin32 module missing. Please install it.")
                
            raise Exception(f"Scanning failed: {e}")

    def mock_scan(self, filename: str, source_path: str) -> str:
        """
        Mocks a scan by copying an existing image to the scan directory.
        Useful for development without a physical scanner.
        """
        import shutil
        output_path = os.path.join(self.scan_dir, filename)
        shutil.copy(source_path, output_path)
        return output_path
