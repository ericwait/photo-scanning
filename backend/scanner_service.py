import os
import subprocess
from typing import Optional
try:
    import twain
except ImportError:
    twain = None

class ScannerService:
    def __init__(self, scan_dir: str):
        self.scan_dir = scan_dir
        if not os.path.exists(scan_dir):
            os.makedirs(scan_dir)

    def scan_page(self, filename: str, dpi: int = 400, bit_depth: int = 24) -> str:
        """
        Triggers a scan using TWAIN (File Transfer).
        Falls back to WIA if TWAIN is not available or fails (optional, but requested to switch).
        """
        if not twain:
            raise Exception("TWAIN module not found. Please install pytwain to scan.")

        output_path = os.path.join(self.scan_dir, filename)
        
        try:
            # 1. Initialize Source Manager
            # Parent window 0.
            sm = twain.SourceManager(0)
            
            # 2. Open Source
            # OpenSource() opens the default source. 
            # To avoid "Select Scanner" popup if no default, we try to open known Epson names directly.
            # 'GetDataSources' was invalid.
            source_name = "EPSON Perfection V600"
            
            try:
                # Try specific first
                print(f"Attempting to open TWAIN Source: {source_name}")
                ss = sm.OpenSource(source_name)
            except:
                ss = None
            
            if not ss:
                print("Specific source not found, trying default...")
                ss = sm.OpenSource()
            
            if not ss:
                raise Exception("No TWAIN source found (or user cancelled selection).")

            print(f"Connected to TWAIN Source: {ss.GetIdentity()['ProductName']}")

            # 3. Configure
            # Resolution
            try:
                ss.SetCapability(twain.ICAP_XRESOLUTION, twain.TWTY_FIX32, float(dpi))
                ss.SetCapability(twain.ICAP_YRESOLUTION, twain.TWTY_FIX32, float(dpi))
            except Exception as e:
                print(f"Warning: Failed to set TWAIN resolution: {e}")

            # Scan Area / Size
            # Ensure we scan the full bed (Max Size / A4 297mm) to avoid cropping
            try:
                # Try Max Size to get the full glass area
                ss.SetCapability(twain.ICAP_SUPPORTEDSIZES, twain.TWTY_UINT16, twain.TWSS_MAXSIZE)
                print("Set scan area to Max Size (Full Bed)")
            except Exception as e:
                # Fallback to A4 (297mm) if Max Size fails
                try:
                    ss.SetCapability(twain.ICAP_SUPPORTEDSIZES, twain.TWTY_UINT16, twain.TWSS_A4)
                    print("Set scan area to A4 (297mm height)")
                except:
                    print(f"Warning: Failed to set scan area size: {e}")

            # Pixel Type (Color)
            try:
                ss.SetCapability(twain.ICAP_PIXELTYPE, twain.TWTY_UINT16, twain.TWPT_RGB)
            except Exception as e:
                 print(f"Warning: Failed to set TWAIN pixel type: {e}")

            # Bit Depth
            # ICAP_BITDEPTH: Try 48 (Total) first, then 16 (Per Channel).
            # Some drivers want total bits (48), some want per-channel (16).
            target_depth_total = 48
            target_depth_channel = 16
            
            if bit_depth == 48:
                depth_set = False
                # Try 48
                try:
                    ss.SetCapability(twain.ICAP_BITDEPTH, twain.TWTY_UINT16, target_depth_total)
                    print(f"Set TWAIN bit depth to {target_depth_total}")
                    depth_set = True
                except:
                    pass
                
                # Try 16 if 48 failed
                if not depth_set:
                    try:
                        ss.SetCapability(twain.ICAP_BITDEPTH, twain.TWTY_UINT16, target_depth_channel)
                        print(f"Set TWAIN bit depth to {target_depth_channel}")
                        depth_set = True
                    except Exception as e:
                        print(f"Warning: Failed to set TWAIN bit depth (tried 48 and 16): {e}")

            # 4. File Transfer Setup
            # Need to set Transfer Mechanism to File (TWSX_FILE = 1)
            try:
                ss.SetCapability(twain.ICAP_XFERMECH, twain.TWTY_UINT16, twain.TWSX_FILE)
            except:
                print("Warning: Could not set transfer mechanism to FILE.")

            # We want TIFF for 48-bit transfer
            try:
                ss.SetCapability(twain.ICAP_IMAGEFILEFORMAT, twain.TWTY_UINT16, twain.TWFF_TIFF)
            except Exception as e:
                print("Warning: Failed to set TWAIN file format to TIFF.")
            
            # Set the filename using low-level DS entry
            # We need a temporary TIFF file for the transfer
            temp_tiff = os.path.join(self.scan_dir, "temp_scan.tif")
            if os.path.exists(temp_tiff):
                os.remove(temp_tiff)

            # Set the filename using standard helper or low-level entry
            # We need a temporary TIFF file for the transfer
            temp_tiff = os.path.join(self.scan_dir, "temp_scan.tif")
            if os.path.exists(temp_tiff):
                os.remove(temp_tiff)
                
            file_setup_ok = False
            
            # 1. Try SetXferFileName helper
            try:
                # Based on error, this method takes (filename, format)
                ss.SetXferFileName(temp_tiff, twain.TWFF_TIFF)
                print(f"Set transfer filename via SetXferFileName: {temp_tiff}")
                file_setup_ok = True
            except AttributeError:
                 # Method might use different casing or args
                 pass
            except Exception as e:
                 print(f"Warning: SetXferFileName helper failed: {e}")

            # 2. Try Low Level via Source Manager if helper missing
            if not file_setup_ok:
                # Since SetXferFileName exists (seen in traceback), we shouldn't reach here usually.
                # But just in case, note that 'sm.ds_entry' might not exist in this wrapper.
                print("Warning: Could not set filename via helper. DSM entry fallback not supported in this wrapper.")

            if not file_setup_ok:
                 print("Critical Warning: Could not set TWAIN file destination. Scanner may save to default location.")

            # Try to suppress progress indicator (if supported)
            try:
                ss.SetCapability(twain.ICAP_INDICATORS, twain.TWTY_BOOL, False)
            except:
                pass

            # 5. Acquire
            ss.RequestAcquire(0, 0)
            
            # 6. Transfer Loop
            infos = ss.GetImageInfo()
            print(f"Acquiring image: {infos}")
            
            # Use XferImageByFile (no args, relies on SETUPFILEXFER)
            ss.XferImageByFile()
            
            print(f"TWAIN temp scan saved to {temp_tiff}")
            
            # Close Source/Manager before processing to free device


            ss.destroy()
            sm.destroy()
            
            # Move the temp TIFF to the requested output path
            # Since we now use TIFF as default, we can just move it.
            if os.path.exists(temp_tiff):
                # If target is also tiff/tif, just move
                if output_path.lower().endswith(('.tiff', '.tif')):
                    import shutil
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    shutil.move(temp_tiff, output_path)
                    print(f"Moved temp TIFF to {output_path}")
                else:
                    # Conversion needed (e.g. if legacy BMP requested)
                    import cv2
                    img = cv2.imread(temp_tiff, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
                    if img is None:
                        raise Exception("Failed to read temp TIFF file")
                    cv2.imwrite(output_path, img) 
                    os.remove(temp_tiff)
                    print(f"Converted and saved scan to {output_path}")
                
                return output_path
            else:
                raise Exception("Temp TIFF file was not created by scanner.")

        except Exception as e:
            print(f"TWAIN Scan Failed: {e}")
            # Ensure cleanup happens even on failure
            try:
                if 'ss' in locals() and ss: ss.destroy()
                if 'sm' in locals() and sm: sm.destroy()
            except:
                pass
            
            # Re-raise error to stop execution
            raise Exception(f"TWAIN Scan failed: {e}")

    def scan_page_wia(self, filename: str, dpi: int = 400, bit_depth: int = 24) -> str:
        """
        Triggers a scan from the default scanner using Windows Image Acquisition (WIA) automation.
        Bypasses the "Select Device" and "Scan Properties" UI prompts by connecting directly.
        """
        # Ensure we use PNG for 48-bit to avoid issues, though BMP might work.
        if bit_depth == 48 and filename.endswith(".bmp"):
            filename = filename.replace(".bmp", ".png")
            
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
            # 4104: Bits Per Pixel (Depth) - 24 or 48
            
            # Force Color Intent (6146)
            try:
                for prop in item.Properties:
                    if prop.PropertyID == 6146: 
                        prop.Value = 1 # Color
                        break
            except Exception as e:
                print(f"Warning: Failed to set Color Intent: {e}")

            # Set DPI (6147/6148)
            try:
                for prop in item.Properties:
                    if prop.PropertyID == 6147: # Horizontal Res
                        prop.Value = dpi
                    elif prop.PropertyID == 6148: # Vertical Res
                        prop.Value = dpi
            except Exception as e:
                print(f"Warning: Failed to set DPI: {e}")

            # Set Bit Depth (4104)
            try:
                if bit_depth == 48:
                     # For 48-bit, we might need to set Data Type (4103) to 0 (Custom) or ensure it allows it.
                     # But primarily we set Depth (4104).
                     depth_set = False
                     for prop in item.Properties:
                        if prop.PropertyID == 4104: # Depth
                             prop.Value = 48
                             depth_set = True
                             break
                     if not depth_set:
                         print("Warning: Scanner does not support setting Bit Depth (Property 4104 not found).")
                else:
                     # Enforce 24-bit logic if needed
                     for prop in item.Properties:
                         if prop.PropertyID == 4104: 
                             prop.Value = 24
            except Exception as e:
                print(f"Warning: Failed to set Bit Depth to {bit_depth}-bit: {e}")
                print("Scanner will likely use default depth (24-bit).")
                        
            print(f"Scanner configured (attempted): Color, {dpi} DPI, {bit_depth}-bit")

            # 4. Transfer the image
            print("Scanning in progress (UI suppressed)...")
            # FormatID for BMP: {B96B3CAB-0728-11D3-9D7B-0000F81EF32E}
            # FormatID for PNG: {B96B3CAF-0728-11D3-9D7B-0000F81EF32E}
            
            # Start with default BMP
            format_id = "{B96B3CAB-0728-11D3-9D7B-0000F81EF32E}" 
            final_ext = ".bmp"
            
            # Use PNG if 48-bit was requested AND successfully configured (or if we just want PNG)
            # Actually, to be safe: If 48-bit failed, we are doing 24-bit.
            # But the caller (main.py) expects .png if it passed 48...
            # Let's check if the filename ends with .png. If so, try to match format.
            if filename.lower().endswith(".png"):
                format_id = "{B96B3CAF-0728-11D3-9D7B-0000F81EF32E}" # PNG
                final_ext = ".png"
            else:
                 final_ext = ".bmp"
                 
            # Note: If the driver ignores the format ID and sends BMP data anyway,
            # we might end up with a .png file containing BMP data.
            # We can rely on image_file.FileExtension if we wanted, but we are saving to `output_path`.
            
            image_file = item.Transfer(format_id)
            
            if image_file:
                # Ensure output filename matches usage
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
