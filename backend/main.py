from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
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
    sensitivity: int = 210
    crop_margin: int = 10
    contrast: float = 1.0
    auto_wb: bool = False

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/images")
async def get_image(path: str):
    if not os.path.exists(path):
        raise HTTPException(404, "Image not found")
    return FileResponse(path)

@app.post("/scan")
async def trigger_scan(request: ScanRequest):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"scan_{timestamp}.bmp"
    
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
        cropped_paths = processor.detect_and_crop(
            scan_path, 
            output_subfolder=request.album_name,
            sensitivity=request.sensitivity,
            crop_margin=request.crop_margin,
            contrast=request.contrast,
            auto_wb=request.auto_wb
        )
        
        # Modify returned paths to be URLs
        processed_urls = []
        for p in cropped_paths:
            # If subfolder is used, p is absolute path. We need relative URL.
            # Assuming OUTPUT_DIR is the root served at /output
            
            # Check if it is inside OUTPUT_DIR
            rel_path = None
            try:
                if os.path.commonpath([p, OUTPUT_DIR]) == OUTPUT_DIR:
                     rel_path = os.path.relpath(p, OUTPUT_DIR)
            except:
                pass # Different drives or paths

            if rel_path:
                 # Ensure forward slashes for URL
                 url_path = f"/output/{rel_path}".replace("\\", "/")
                 processed_urls.append(url_path)
            else:
                 # Absolute path outside of project structure
                 url_path = f"/images?path={p}"
                 processed_urls.append(url_path)
    
        return {
            "status": "success",
            "scan_path": f"/scans/{filename}",
            "photos": processed_urls
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class RefineRequest(BaseModel):
    scan_path: str
    photo_index: int
    points: list[list[int]] # [[x,y], [x,y], [x,y], [x,y]]
    album_name: str = "default"

@app.post("/refine")
async def refine_photo(request: RefineRequest):
    try:
        # scan_path comes as "/scans/filename.jpg", we need absolute path
        # Assume scan is always in SCAN_DIR for now (managed by us)
        filename = os.path.basename(request.scan_path)
        filepath = os.path.join(SCAN_DIR, filename)
        
        output_path = processor.manual_crop(filepath, request.points, request.photo_index, output_subfolder=request.album_name)
        
        # Determine strict URL return
        url_path = f"/images?path={output_path}" # Fallback
        try:
             # Try standard relative first
             if os.path.commonpath([output_path, OUTPUT_DIR]) == OUTPUT_DIR:
                rel = os.path.relpath(output_path, OUTPUT_DIR)
                url_path = f"/output/{rel}".replace("\\", "/")
        except:
             pass

        return {
            "status": "success",
            "photo_url": url_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class RotateRequest(BaseModel):
    photo_url: str
    angle: int # 90 or -90

@app.post("/rotate")
async def rotate_photo(request: RotateRequest):
    try:
        # Decode photo_url to absolute path
        p = request.photo_url
        abs_path = ""
        
        if p.startswith("/output/"):
            # Relative to output dir. 
            # p might be /output/subfolder/file.png OR /output/file.png
            # Strip /output/
            rel = p[len("/output/"):]
            abs_path = os.path.join(OUTPUT_DIR, rel)
        elif "/images?path=" in p:
             # Extract path param
             # Simple string split? "path="
             abs_path = p.split("path=")[1]
             # If url encoded? Frontend might send raw.
             # Assuming standard string for now.
        else:
             # Fallback, maybe it's just a filename?
             abs_path = os.path.join(OUTPUT_DIR, os.path.basename(p))

        processor.rotate_photo(abs_path, request.angle)
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
