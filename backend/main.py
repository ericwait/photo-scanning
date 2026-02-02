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
    auto_contrast: bool = False
    auto_wb: bool = False
    dpi: int = 400
    bit_depth: int = 24
    grid_rows: int = 3
    grid_cols: int = 1
    ignore_black_background: bool = False

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
    filename = f"scan_{timestamp}.tiff"
    
    try:
        if request.mock:
            if not request.mock_source:
                raise HTTPException(status_code=400, detail="Mock source path required for mock scan")
            # Resolve mock source relative to project root (BASE_DIR)
            mock_source_path = os.path.join(BASE_DIR, request.mock_source)
            scan_path = scanner.mock_scan(filename, mock_source_path)
        else:
            scan_path = scanner.scan_page(filename, dpi=request.dpi, bit_depth=request.bit_depth)
        
        # Determine actual filename used (in case scanner_service changed extension)
        actual_filename = os.path.basename(scan_path)
        scan_url_path = f"/scans/{actual_filename}"

        # If 48-bit OR TIFF, browsers often can't display it.
        # Create an 8-bit PNG copy for the UI preview.
        # We also might have switched to PNG from BMP.
        is_tiff = scan_path.lower().endswith(('.tiff', '.tif'))
        if request.bit_depth == 48 or is_tiff:
            import cv2
            import numpy as np
            
            # Read the scan (16-bit or TIFF)
            # We must verify it exists first? (scanner scan_path ensures it)
            # Use ANYDEPTH to preserve 16-bit if present, but convert to 8-bit for preview
            img = cv2.imread(scan_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            
            if img is not None:
                # Create 8-bit preview
                # Change extension to .png for preview
                preview_filename = f"preview_{os.path.splitext(actual_filename)[0]}.png"
                preview_path = os.path.join(SCAN_DIR, preview_filename)
                
                # Convert to 8-bit if needed
                if img.dtype == np.uint16:
                    img_8 = (img / 256.0).astype(np.uint8)
                else:
                    img_8 = img
                
                cv2.imwrite(preview_path, img_8)
                
                # Use this for the frontend
                scan_url_path = f"/scans/{preview_filename}"

        # Process the scan immediately
        results = processor.detect_and_crop(
            scan_path, 
            output_subfolder=request.album_name,
            sensitivity=request.sensitivity,
            crop_margin=request.crop_margin,
            contrast=request.contrast,
            auto_contrast=request.auto_contrast,
            auto_wb=request.auto_wb,
            grid_rows=request.grid_rows,
            grid_cols=request.grid_cols,
            ignore_black_background=request.ignore_black_background,
            dpi=request.dpi
        )
        
        # Modify returned paths to be URLs
        processed_photos = []
        for item in results:
            p = item["path"]
            points = item["points"]
            
            # If subfolder is used, p is absolute path. We need relative URL.
            # Assuming OUTPUT_DIR is the root served at /output
            
            # Check if it is inside OUTPUT_DIR
            rel_path = None
            try:
                if os.path.commonpath([p, OUTPUT_DIR]) == OUTPUT_DIR:
                     rel_path = os.path.relpath(p, OUTPUT_DIR)
            except:
                pass # Different drives or paths

            url_path = ""
            if rel_path:
                 # Ensure forward slashes for URL
                 url_path = f"/output/{rel_path}".replace("\\", "/")
            else:
                 # Absolute path outside of project structure
                 url_path = f"/images?path={p}"
            
            processed_photos.append({
                "url": url_path,
                "points": points
            })
        return {
            "status": "success",
            "scan_path": scan_url_path,
            "photos": processed_photos
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class RefineRequest(BaseModel):
    scan_path: str
    photo_index: int
    points: list[list[int]] # [[x,y], [x,y], [x,y], [x,y]]
    album_name: str = "default"
    contrast: float = 1.0
    auto_contrast: bool = False
    auto_wb: bool = False

@app.post("/refine")
async def refine_photo(request: RefineRequest):
    try:
        # scan_path comes as "/scans/filename.jpg" or "/scans/preview_filename.png"
        filename = os.path.basename(request.scan_path)
        
        # If the frontend is sending the preview path, we want to solve relevant to the MASTER scan (TIFF)
        # Check if it starts with "preview_"
        if filename.startswith("preview_"):
            real_filename = filename.replace("preview_", "")
            # The preview is .png, but master is likely .tiff or .bmp
            # Try to find the master file
            potential_master = os.path.join(SCAN_DIR, real_filename)
            
            # If extension is still .png, change to .tiff?
            if not os.path.exists(potential_master):
                name, ext = os.path.splitext(real_filename)
                # Try .tiff
                potential_master_tiff = os.path.join(SCAN_DIR, f"{name}.tiff")
                if os.path.exists(potential_master_tiff):
                     filepath = potential_master_tiff
                elif os.path.exists(os.path.join(SCAN_DIR, f"{name}.bmp")):
                     filepath = os.path.join(SCAN_DIR, f"{name}.bmp")
                else:
                     # Fallback to the preview file itself if master lost?
                     filepath = os.path.join(SCAN_DIR, filename)
            else:
                filepath = potential_master
        else:
            filepath = os.path.join(SCAN_DIR, filename)
            
        print(f"Refining from source: {filepath}")
        
        output_path = processor.manual_crop(
            filepath, 
            request.points, 
            request.photo_index, 
            output_subfolder=request.album_name,
            contrast=request.contrast,
            auto_contrast=request.auto_contrast,
            auto_wb=request.auto_wb
        )
        
        # Determine strict URL return
        url_path = f"/images?path={output_path}" # Fallback
        try:
             # Try standard relative first
             if os.path.commonpath([output_path, OUTPUT_DIR]) == OUTPUT_DIR:
                rel = os.path.relpath(output_path, OUTPUT_DIR)
                url_path = f"/output/{rel}".replace("\\", "/")
        except:
             pass

        # Append timestamp to force browser cache refresh
        import time
        ts = int(time.time())
        if "?" in url_path:
             url_path += f"&t={ts}"
        else:
             url_path += f"?t={ts}"

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
            
            # Strip query params
            clean_p = p
            if "?" in clean_p:
                clean_p = clean_p.split("?")[0]
            
            # Strip /output/
            rel = clean_p[len("/output/"):]
            abs_path = os.path.join(OUTPUT_DIR, rel)
        elif "/images?path=" in p:
             # Extract path param
             # p is like /images?path=C:\Users\foo.png&t=1234
             temp = p.split("path=")[1]
             # Strip additional query params if any
             if "&" in temp:
                 temp = temp.split("&")[0]
             abs_path = temp
        else:
             # Fallback, maybe it's just a filename?
             # Clean p of query params just in case
             if "?" in p:
                 p = p.split("?")[0]
             abs_path = os.path.join(OUTPUT_DIR, os.path.basename(p))

        processor.rotate_photo(abs_path, request.angle)
        
        # Return the URL with a fresh timestamp so the UI updates
        clean_url = request.photo_url.split('?')[0].split('&')[0]
        import time
        new_url = f"{clean_url}?t={int(time.time())}"
        
        return {
            "status": "success",
            "photo_url": new_url
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
    uvicorn.run(app, host="0.0.0.0", port=8002)
