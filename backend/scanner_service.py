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
        """
        output_path = os.path.join(self.scan_dir, filename)
        
        try:
            import win32com.client
            
            # WIA Common Dialog constants
            WIA_DEVICE_DIALOG_SINGLE_IMAGE = 1
            WIA_INTENT_IMAGE_TYPE_COLOR = 1
            
            # Create the WIA Common Dialog object
            # This allows specificying intent and getting the image easily
            wia_dialog = win32com.client.Dispatch("WIA.CommonDialog")
            
            # ShowDeviceDialog might show a UI selection if multiple scanners, 
            # or if parameters need confirming. 
            # To be fully automated, we might need a lower level approach, 
            # but CommonDialog is the standard way to get an ImageFile over WIA.
            # Using ShowAcquireImage(DeviceType=1 (Scanner), Intent=1 (Color), Bias=65536 (Maximize Quality), Format="{B96B3CAE-0728-11D3-9D7B-0000F81EF32E}" (JPEG))
            
            # Note: 65536 is "Maximize Quality" bias
            # FormatID for JPEG is {B96B3CAE-0728-11D3-9D7B-0000F81EF32E}
            
            print("Requesting scan from WIA device...")
            image_file = wia_dialog.ShowAcquireImage(
                1, # DeviceType: Scanner
                1, # Intent: ColorIntent
                65536, # Bias: Maximize Quality
                "{B96B3CAE-0728-11D3-9D7B-0000F81EF32E}", # FormatID: JPEG
                False, # AlwaysSelectDevice: False (use default if possible)
                True, # UseCommonUI: True (shows progress bar, helpful)
                False # CancelError: False
            )
            
            if image_file:
                # SaveFile expects a filename without extension if it manages formatting,
                # but image_file.SaveFile(path) is standard.
                # WIA ImageFile SaveFile method.
                if os.path.exists(output_path):
                    os.remove(output_path)
                    
                image_file.SaveFile(output_path)
                print(f"Scan saved to {output_path}")
                return output_path
            else:
                raise Exception("No image returned from scanner.")
                
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
