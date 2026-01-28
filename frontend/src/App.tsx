import { useState, useEffect } from 'react'
import { RefineModal } from './RefineModal';

import { PhotoCard, type PhotoData } from './PhotoCard';

interface ScanResult {
  status: string;
  scan_path: string;
  photos: PhotoData[];
}

function App() {
  const [isScanning, setIsScanning] = useState(false);
  const [currentScan, setCurrentScan] = useState<ScanResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Refine Modal State
  const [isRefineOpen, setIsRefineOpen] = useState(false);
  const [refinePhotoIndex, setRefinePhotoIndex] = useState<number>(-1);

  // Scan Settings
  // Scan Settings - Initialize from localStorage if available
  const [albumName, setAlbumName] = useState(() => localStorage.getItem('albumName') || "Default");
  const [sensitivity, setSensitivity] = useState(() => Number(localStorage.getItem('sensitivity')) || 210);
  const [cropMargin, setCropMargin] = useState(() => Number(localStorage.getItem('cropMargin')) || 10);
  const [contrast, setContrast] = useState(() => {
    const saved = localStorage.getItem('contrast');
    return saved ? Number(saved) : 1.0;
  });
  const [autoContrast, setAutoContrast] = useState(() => localStorage.getItem('autoContrast') === 'true');
  const [autoWb, setAutoWb] = useState(() => localStorage.getItem('autoWb') === 'true');
  const [dpi, setDpi] = useState(() => Number(localStorage.getItem('dpi')) || 400);
  const [use48Bit, setUse48Bit] = useState(() => localStorage.getItem('use48Bit') === 'true');
  const [showSettings, setShowSettings] = useState(() => localStorage.getItem('showSettings') === 'true');
  const [gridRows, setGridRows] = useState(() => Number(localStorage.getItem('gridRows')) || 3);
  const [gridCols, setGridCols] = useState(() => Number(localStorage.getItem('gridCols')) || 1);
  const [ignoreBlackBackground, setIgnoreBlackBackground] = useState(() => localStorage.getItem('ignoreBlackBackground') === 'true');

  // Persist settings changes
  useEffect(() => { localStorage.setItem('albumName', albumName); }, [albumName]);
  useEffect(() => { localStorage.setItem('sensitivity', String(sensitivity)); }, [sensitivity]);
  useEffect(() => { localStorage.setItem('cropMargin', String(cropMargin)); }, [cropMargin]);
  useEffect(() => { localStorage.setItem('contrast', String(contrast)); }, [contrast]);
  useEffect(() => { localStorage.setItem('autoContrast', String(autoContrast)); }, [autoContrast]);
  useEffect(() => { localStorage.setItem('autoWb', String(autoWb)); }, [autoWb]);
  useEffect(() => { localStorage.setItem('dpi', String(dpi)); }, [dpi]);
  useEffect(() => { localStorage.setItem('use48Bit', String(use48Bit)); }, [use48Bit]);
  useEffect(() => { localStorage.setItem('showSettings', String(showSettings)); }, [showSettings]);
  useEffect(() => { localStorage.setItem('gridRows', String(gridRows)); }, [gridRows]);
  useEffect(() => { localStorage.setItem('gridCols', String(gridCols)); }, [gridCols]);
  useEffect(() => { localStorage.setItem('ignoreBlackBackground', String(ignoreBlackBackground)); }, [ignoreBlackBackground]);

  const API_BASE = "http://localhost:8001";

  // Removed fetchHistory as history panel is gone


  const handleScan = async (mock = false) => {
    setIsScanning(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/scan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mock: mock,
          mock_source: mock ? "test_scanner_bed.png" : "",
          album_name: albumName,
          sensitivity: sensitivity,
          crop_margin: cropMargin,
          contrast: contrast,
          auto_contrast: autoContrast,
          auto_wb: autoWb,
          dpi: dpi,
          bit_depth: use48Bit ? 48 : 24,
          grid_rows: gridRows,
          grid_cols: gridCols,
          ignore_black_background: ignoreBlackBackground
        })
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Scan failed");
      }

      const data: ScanResult = await response.json();
      setCurrentScan(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsScanning(false);
    }
  };

  const openRefine = (index: number) => {
    setRefinePhotoIndex(index);
    setIsRefineOpen(true);
  };

  const handleRefineSaveInternal = async (index: number, points: number[][], settings?: { contrast: number; autoContrast: boolean; autoWb: boolean }) => {
    if (!currentScan) return;

    // Use passed settings or fallback to global state
    const useContrast = settings ? settings.contrast : contrast;
    const useAutoContrast = settings ? settings.autoContrast : autoContrast;
    const useAutoWb = settings ? settings.autoWb : autoWb;

    try {
      const response = await fetch(`${API_BASE}/refine`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scan_path: currentScan.scan_path,
          photo_index: index,
          points: points,
          album_name: albumName,
          contrast: useContrast,
          auto_contrast: useAutoContrast,
          auto_wb: useAutoWb
        })
      });

      if (!response.ok) {
        throw new Error("Failed to refine photo");
      }

      const data = await response.json();

      // Update the local state with the new photo URL to reflect the change immediately
      const separator = data.photo_url.includes('?') ? '&' : '?';
      const newPhotoUrl = `${data.photo_url}${separator}t=${Date.now()}`;

      const updatedPhotos = [...currentScan.photos];
      // Keep existing points, update URL
      updatedPhotos[index] = { ...updatedPhotos[index], url: newPhotoUrl };

      setCurrentScan({
        ...currentScan,
        photos: updatedPhotos
      });

      // Close modal if open and matching index
      if (isRefineOpen && refinePhotoIndex === index) {
        setIsRefineOpen(false);
      }

    } catch (err: any) {
      alert(`Error refining photo: ${err.message}`);
    }
  };

  const handleRefineSave = async (points: number[][], settings?: { contrast: number; autoContrast: boolean; autoWb: boolean }) => {
    await handleRefineSaveInternal(refinePhotoIndex, points, settings);
  };

  const handleRotate = async (index: number, angle: number) => {
    if (!currentScan) return;
    const photoUrl = currentScan.photos[index].url;

    // Remove timestamp param (either ?t=... or &t=...) to get clean ID
    const cleanUrl = photoUrl.replace(/([?&])t=\d+$/, '');

    try {
      const response = await fetch(`${API_BASE}/rotate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          photo_url: cleanUrl,
          angle: angle
        })
      });

      if (!response.ok) throw new Error("Rotate failed");

      // Force refresh with new timestamp
      const separator = cleanUrl.includes('?') ? '&' : '?';
      const newPhotoUrl = `${cleanUrl}${separator}t=${Date.now()}`;
      const updatedPhotos = [...currentScan.photos];
      updatedPhotos[index] = { ...updatedPhotos[index], url: newPhotoUrl };

      setCurrentScan({
        ...currentScan,
        photos: updatedPhotos
      });

    } catch (err: any) {
      alert(err.message);
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 p-8">
      {currentScan && (
        <RefineModal
          isOpen={isRefineOpen}
          onClose={() => setIsRefineOpen(false)}
          onSave={(points, settings) => handleRefineSave(points, settings)}
          imageUrl={`${API_BASE}${currentScan.scan_path}`}
          initialSettings={{
            contrast: contrast,
            autoContrast: autoContrast,
            autoWb: autoWb
          }}
        />
      )}
      <header className="flex justify-between items-center mb-8 bg-slate-800 p-6 rounded-2xl shadow-xl border border-slate-700">
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-emerald-400 bg-clip-text text-transparent">
            Photo Scanning Studio
          </h1>
          <p className="text-slate-400 mt-1">Epson V600 Batch Processor</p>
        </div>
        <div className="flex gap-4">
          <button
            onClick={() => handleScan(true)}
            className="px-6 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-all font-semibold"
            disabled={isScanning}
          >
            Mock Scan
          </button>
          <button
            onClick={() => handleScan(false)}
            className="px-8 py-3 bg-blue-600 hover:bg-blue-500 rounded-xl transition-all font-bold shadow-lg shadow-blue-900/40 disabled:opacity-50"
            disabled={isScanning}
          >
            {isScanning ? "Scanning..." : "Scan New Page"}
          </button>
        </div>
      </header>

      {/* Settings Panel */}
      <div className="mb-8">
        <button
          onClick={() => setShowSettings(!showSettings)}
          className="text-slate-400 hover:text-white flex items-center gap-2 text-sm font-semibold mb-2 transition-colors"
        >
          {showSettings ? "▼ Hide Settings" : "▶ Show Advanced Settings"}
        </button>

        {showSettings && (
          <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 grid grid-cols-1 md:grid-cols-3 gap-6 animate-fade-in-down">
            {/* Grid Layout Settings */}
            <div>
              <div className="flex justify-between mb-2">
                <label className="text-sm font-medium text-slate-400">
                  Expected Grid
                </label>
                <span className="text-xs text-slate-500 bg-slate-900 px-2 py-0.5 rounded">
                  {gridRows} x {gridCols} ({gridRows * gridCols} Photos)
                </span>
              </div>
              <div className="flex gap-4">
                <div className="flex-1">
                  <label className="text-xs text-slate-500 mb-1 block">Rows</label>
                  <input
                    type="number"
                    min="1"
                    max="10"
                    value={gridRows}
                    onChange={(e) => setGridRows(Number(e.target.value))}
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white focus:ring-2 focus:ring-blue-500 focus:outline-none"
                  />
                </div>
                <div className="flex-1">
                  <label className="text-xs text-slate-500 mb-1 block">Cols</label>
                  <input
                    type="number"
                    min="1"
                    max="10"
                    value={gridCols}
                    onChange={(e) => setGridCols(Number(e.target.value))}
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white focus:ring-2 focus:ring-blue-500 focus:outline-none"
                  />
                </div>
              </div>
              <p className="text-xs text-slate-500 mt-1">Arrangement on scanner bed.</p>
            </div>

            {/* Output Folder / Album */}
            <div>
              <label className="block text-sm font-medium text-slate-400 mb-2">
                Output Folder (Absolute Path or Subfolder)
              </label>
              <input
                type="text"
                value={albumName}
                onChange={(e) => setAlbumName(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-white focus:ring-2 focus:ring-blue-500 focus:outline-none"
                placeholder="e.g. Vacation 2024"
              />
            </div>

            {/* Detection Sensitivity */}
            <div>
              <div className="flex justify-between mb-2">
                <label className="text-sm font-medium text-slate-400">
                  Detection Sensitivity (Threshold)
                </label>
                <span className="text-xs text-slate-500 bg-slate-900 px-2 py-0.5 rounded">{sensitivity}</span>
              </div>
              <input
                type="range"
                min="150"
                max="250"
                value={sensitivity}
                onChange={(e) => setSensitivity(Number(e.target.value))}
                className="w-full accent-blue-500 h-2 bg-slate-900 rounded-lg appearance-none cursor-pointer"
              />
              <p className="text-xs text-slate-500 mt-1">Lower = requires darker photos. Higher = picks up more (risk of background).</p>
            </div>

            {/* Crop Margin */}
            <div>
              <div className="flex justify-between mb-2">
                <label className="text-sm font-medium text-slate-400">
                  Crop Tightness (Margin px)
                </label>
                <span className="text-xs text-slate-500 bg-slate-900 px-2 py-0.5 rounded">{cropMargin}px</span>
              </div>
              <input
                type="range"
                min="0"
                max="50"
                value={cropMargin}
                onChange={(e) => setCropMargin(Number(e.target.value))}
                className="w-full accent-emerald-500 h-2 bg-slate-900 rounded-lg appearance-none cursor-pointer"
              />
              <p className="text-xs text-slate-500 mt-1">Pixels to shave off the edges. Increase if seeing white borders.</p>
            </div>

            {/* Contrast Boost */}
            <div>
              <div className="flex justify-between mb-2">
                <label className="text-sm font-medium text-slate-400">
                  Contrast Boost
                </label>
                <span className="text-xs text-slate-500 bg-slate-900 px-2 py-0.5 rounded">{contrast}x</span>
              </div>
              <input
                type="range"
                min="1.0"
                max="2.0"
                step="0.1"
                value={contrast}
                onChange={(e) => setContrast(Number(e.target.value))}
                className="w-full accent-purple-500 h-2 bg-slate-900 rounded-lg appearance-none cursor-pointer"
              />
              <p className="text-xs text-slate-500 mt-1">Increase to make faded colors pop.</p>
            </div>

            {/* DPI Settings */}
            <div>
              <div className="flex justify-between mb-2">
                <label className="text-sm font-medium text-slate-400">
                  Scan Resolution (DPI)
                </label>
                <span className="text-xs text-slate-500 bg-slate-900 px-2 py-0.5 rounded">{dpi} DPI</span>
              </div>
              <input
                type="range"
                min="150"
                max="1200"
                step="50"
                value={dpi}
                onChange={(e) => setDpi(Number(e.target.value))}
                className="w-full accent-pink-500 h-2 bg-slate-900 rounded-lg appearance-none cursor-pointer"
              />
              <p className="text-xs text-slate-500 mt-1">Higher = more detail but slower. 300-600 is standard.</p>
            </div>

            {/* Auto White Balance & Contrast & 48-bit & BlackBg */}
            <div className="flex flex-col gap-4">
              <div>
                <label className="flex items-center gap-3 cursor-pointer group">
                  <div className={`w-6 h-6 rounded border flex items-center justify-center transition-colors ${autoWb ? 'bg-blue-600 border-blue-600' : 'bg-slate-900 border-slate-700'}`}>
                    {autoWb && <span className="text-white text-sm">✓</span>}
                  </div>
                  <input
                    type="checkbox"
                    className="hidden"
                    checked={autoWb}
                    onChange={(e) => setAutoWb(e.target.checked)}
                  />
                  <div>
                    <span className="block text-sm font-medium text-slate-300 group-hover:text-white transition-colors">Aggressive Auto WB</span>
                    <p className="text-xs text-slate-500 mt-1">Normalize color channels (removes casts).</p>
                  </div>
                </label>
              </div>

              <div>
                <label className="flex items-center gap-3 cursor-pointer group">
                  <div className={`w-6 h-6 rounded border flex items-center justify-center transition-colors ${autoContrast ? 'bg-indigo-600 border-indigo-600' : 'bg-slate-900 border-slate-700'}`}>
                    {autoContrast && <span className="text-white text-sm">✓</span>}
                  </div>
                  <input
                    type="checkbox"
                    className="hidden"
                    checked={autoContrast}
                    onChange={(e) => setAutoContrast(e.target.checked)}
                  />
                  <div>
                    <span className="block text-sm font-medium text-slate-300 group-hover:text-white transition-colors">Auto Contrast</span>
                    <p className="text-xs text-slate-500 mt-1">Stretch min/max values to full range.</p>
                  </div>
                </label>
              </div>

              <div>
                <label className="flex items-center gap-3 cursor-pointer group">
                  <div className={`w-6 h-6 rounded border flex items-center justify-center transition-colors ${use48Bit ? 'bg-yellow-600 border-yellow-600' : 'bg-slate-900 border-slate-700'}`}>
                    {use48Bit && <span className="text-white text-sm">✓</span>}
                  </div>
                  <input
                    type="checkbox"
                    className="hidden"
                    checked={use48Bit}
                    onChange={(e) => setUse48Bit(e.target.checked)}
                  />
                  <div>
                    <span className="block text-sm font-medium text-slate-300 group-hover:text-white transition-colors">High Quality Scan (48-bit)</span>
                    <p className="text-xs text-slate-500 mt-1">Slower. Uses 16-bits per color for smoother gradients.</p>
                  </div>
                </label>
              </div>

              <div>
                <label className="flex items-center gap-3 cursor-pointer group">
                  <div className={`w-6 h-6 rounded border flex items-center justify-center transition-colors ${ignoreBlackBackground ? 'bg-gray-600 border-gray-600' : 'bg-slate-900 border-slate-700'}`}>
                    {ignoreBlackBackground && <span className="text-white text-sm">✓</span>}
                  </div>
                  <input
                    type="checkbox"
                    className="hidden"
                    checked={ignoreBlackBackground}
                    onChange={(e) => setIgnoreBlackBackground(e.target.checked)}
                  />
                  <div>
                    <span className="block text-sm font-medium text-slate-300 group-hover:text-white transition-colors">Ignore Black Background</span>
                    <p className="text-xs text-slate-500 mt-1">Use if scanning on black mat. Ignores very dark areas.</p>
                  </div>
                </label>
              </div>
            </div>
          </div>
        )}
      </div>

      {
        error && (
          <div className="bg-red-500/10 border border-red-500/50 text-red-400 p-4 rounded-xl mb-8">
            Error: {error}
          </div>
        )
      }

      <main className="grid grid-cols-1 lg:grid-cols-2 gap-8 max-w-7xl mx-auto">
        {/* Left: Previous Scan Preview */}
        <div className="space-y-6">
          <div className="bg-slate-800 p-6 rounded-2xl border border-slate-700 h-fit sticky top-8">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <span className="w-2 h-6 bg-blue-500 rounded-full"></span>
              Full Page Scan
            </h2>
            {currentScan ? (
              <div className="relative group overflow-hidden rounded-xl border border-slate-700">
                <img
                  src={`${API_BASE}${currentScan.scan_path}`}
                  alt="Full Scan"
                  className="w-full h-auto"
                />
                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                  <span className="bg-white/10 backdrop-blur-md px-4 py-2 rounded-full text-sm">Preview of scanner bed</span>
                </div>
              </div>
            ) : (
              <div className="aspect-[3/4] bg-slate-900/50 rounded-xl border-2 border-dashed border-slate-700 flex flex-col items-center justify-center text-slate-500">
                <p>No scan yet. Place a page and click "Scan New Page"</p>
              </div>
            )}
          </div>
        </div>

        {/* Right: Detected Photos */}
        <div className="space-y-6">
          <div className="bg-slate-800 p-6 rounded-2xl border border-slate-700">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <span className="w-2 h-6 bg-emerald-500 rounded-full"></span>
              Detected Photos
            </h2>
            <div className="space-y-8">
              {currentScan?.photos.map((photo, idx) => (
                <PhotoCard
                  key={idx}
                  index={idx}
                  photo={photo}
                  onRotate={handleRotate}
                  onRefineOpen={openRefine}
                  onUpdateSettings={(idx, photo, settings) => {
                    // Update the state so the API call uses the correct index (refinePhotoIndex needs to be set possibly, OR we pass index directly)
                    // handleRefineSave uses refinePhotoIndex state. We should probably update it or refactor handleRefineSave to take index.
                    // Let's refactor handleRefineSave to accept index.
                    // Wait, I cannot refactor definitions easily in multi_replace chunks if they are far apart.
                    // I will set state then call. But state setting is async.
                    // Better to just call a specific function.

                    // Actually, handleRefineSave depends on `refinePhotoIndex`.
                    // I should modify handleRefineSave to accept an optional index argument.
                    setRefinePhotoIndex(idx); // This might not update in time for the current function execution?
                    // Actually, we can just copy the logic or modify handleRefineSave.
                    // Let's modify handleRefineSave to take explicit index.
                    handleRefineSaveInternal(idx, photo.points, settings);
                  }}
                  globalSettings={{
                    contrast,
                    autoContrast,
                    autoWb
                  }}
                  scanId={currentScan.scan_path}
                />
              ))}
              {!currentScan && (
                <p className="text-slate-500 italic text-center py-8">Photos will appear here after scanning</p>
              )}
            </div>
          </div>
        </div>
      </main>
    </div >
  )
}

export default App
