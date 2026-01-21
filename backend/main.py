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

# Services
# Define base directory relative to this file to handle being run from different locations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN_DIR = os.path.join(BASE_DIR, "scans")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

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
    filename = f"scan_{timestamp}.png"
    
    try:
        if request.mock:
            if not request.mock_source:
                raise HTTPException(status_code=400, detail="Mock source path required for mock scan")
            # Resolve mock source relative to project root (BASE_DIR)
            mock_source_path = os.path.join(BASE_DIR, request.mock_source)
            scan_path = scanner.mock_scan(filename, mock_source_path)
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

class RefineRequest(BaseModel):
    scan_path: str
    photo_index: int
    points: list[list[int]] # [[x,y], [x,y], [x,y], [x,y]]

@app.post("/refine")
async def refine_photo(request: RefineRequest):
    try:
        # scan_path comes as "/scans/filename.jpg", we need absolute path
        filename = os.path.basename(request.scan_path)
        filepath = os.path.join(SCAN_DIR, filename)
        
        output_path = processor.manual_crop(filepath, request.points, request.photo_index)
        
        return {
            "status": "success",
            "photo_url": f"/output/{os.path.basename(output_path)}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class RotateRequest(BaseModel):
    photo_url: str
    angle: int # 90 or -90

@app.post("/rotate")
async def rotate_photo(request: RotateRequest):
    try:
        processor.rotate_photo(request.photo_url, request.angle)
        return {
            "status": "success",
            "photo_url": request.photo_url
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
