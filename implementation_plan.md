# Photo Scanning & Post-processing Tool Plan

The goal is to create a simple interface that allows a user to scan a page of photos (typically 3 high on a page), automatically detect, straighten, and crop them, and provide a way to refine the results before saving.

## Proposed Architecture

- **Backend**: Python with FastAPI.
  - `ScannerController`: Interfaces with the Epson V600 via TWAIN (best for quality/control).
  - `ImageProcessor`: Uses OpenCV to find contours, straighten (deskew), and crop photos.
  - `Storage`: Local file system for raw scans and processed photos.
- **Frontend**: Vite + React + Tailwind CSS.
  - Simple dashboard with "Scan New Page" button.
  - Preview area for the full scan.
  - Gallery of detected/cropped photos with "Refine" options.

## Proposed Changes

### Backend Implementation

#### [NEW] [scannner_service.py](file:///e:/programming/photo-scanning/backend/scanner_service.py)
Logic to interface with the scanner using `pytwain` or `wia-scan`.

#### [NEW] [processor_service.py](file:///e:/programming/photo-scanning/backend/processor_service.py)
OpenCV logic for:
- Detecting photos on the scanner bed.
- Deskewing (straightening).
- Cropping and color correction.

#### [NEW] [main.py](file:///e:/programming/photo-scanning/backend/main.py)
FastAPI application with endpoints for:
- `/scan`: Trigger a new scan.
- `/process`: Re-run processing on an existing scan with refined parameters.
- `/photos`: List and retrieve processed photos.

### Frontend Implementation

#### [NEW] [frontend/](file:///e:/programming/photo-scanning/frontend/)
A React application to:
- State management for current scan status.
- Display the result of the previous scan.
- Provide a UI to "Refine" crops (e.g., adjusting bounding boxes if auto-detection misses).

## Verification Plan

### Automated Tests
- Mock scanner interface to test processing logic without hardware.
- Test `processor_service.py` with sample "page scans" containing multiple photos.

### Manual Verification
- Run the app and trigger a scan.
- Verify that the Epson V600 initiates a scan.
- Check if the 3 photos on a page are correctly identified and straightened.
- Test the "Refine" flow by manually adjusting a crop in the UI.

> [!IMPORTANT]
> Since I don't have physical access to the scanner, initial development will use "dummy" scan images to test the processing logic. I will need the user to test the actual TWAIN integration once the backend is ready.
