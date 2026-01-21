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
        Triggers a scan from the default scanner and saves it to the scan_dir.
        Uses wia-scan CLI for simplicity in this initial version.
        """
        output_path = os.path.join(self.scan_dir, filename)
        
        # wia-scan command line usage: wia-scan.exe --output "path/to/save.jpg"
        # We assume the user has the scanner connected and it's the default.
        try:
            # Using wia-scan via subprocess if we can't get pytwain to work easily without UI
            # wia-scan is a great alternative for background/scripted scanning.
            result = subprocess.run(
                ["wia-scan", "--output", output_path],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"Scan successful: {result.stdout}")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"Error during scan: {e.stderr}")
            raise Exception(f"Scanning failed: {e.stderr}")
        except FileNotFoundError:
            # If wia-scan CLI is not in PATH, we might need to use the python module directly
            # but wia-scan package usually provides the CLI.
            raise Exception("wia-scan CLI not found. Please ensure wia-scan is installed and in PATH.")

    def mock_scan(self, filename: str, source_path: str) -> str:
        """
        Mocks a scan by copying an existing image to the scan directory.
        Useful for development without a physical scanner.
        """
        import shutil
        output_path = os.path.join(self.scan_dir, filename)
        shutil.copy(source_path, output_path)
        return output_path
