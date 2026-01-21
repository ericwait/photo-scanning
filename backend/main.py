from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import datetime
from scanner_service import ScannerService
from processor_service import ProcessorService

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
SCAN_DIR = "scans"
OUTPUT_DIR = "output"

# Services
scanner = ScannerService(SCAN_DIR)
processor = ProcessorService(OUTPUT_DIR)

# Ensure directories exist
os.makedirs(SCAN_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Static files for previews
app.mount("/scans", StaticFiles(directory=SCAN_DIR), name="scans")
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

class ScanRequest(BaseModel):
    album_name: str = "default"
    mock: bool = False
    mock_source: str = ""

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/scan")
async def trigger_scan(request: ScanRequest):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"scan_{timestamp}.jpg"
    
    try:
        if request.mock:
            if not request.mock_source:
                raise HTTPException(status_code=400, detail="Mock source path required for mock scan")
            scan_path = scanner.mock_scan(filename, request.mock_source)
        else:
            scan_path = scanner.scan_page(filename)
        
        # Process the scan immediately
        cropped_paths = processor.detect_and_crop(scan_path)
        
        return {
            "status": "success",
            "scan_path": f"/scans/{filename}",
            "photos": [f"/output/{os.path.basename(p)}" for p in cropped_paths]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history():
    # Basic history by listing output directory
    photos = [f"/output/{f}" for f in os.listdir(OUTPUT_DIR) if f.endswith((".jpg", ".jpeg", ".png"))]
    return {"photos": sorted(photos, reverse=True)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
