import { useState, useEffect } from 'react'
import { RefineModal } from './RefineModal';

interface ScanResult {
  status: string;
  scan_path: string;
  photos: string[];
}

function App() {
  const [isScanning, setIsScanning] = useState(false);
  const [currentScan, setCurrentScan] = useState<ScanResult | null>(null);
  const [history, setHistory] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Refine Modal State
  const [isRefineOpen, setIsRefineOpen] = useState(false);
  const [refinePhotoIndex, setRefinePhotoIndex] = useState<number>(-1);

  const API_BASE = "http://localhost:8000";

  const fetchHistory = async () => {
    try {
      const response = await fetch(`${API_BASE}/history`);
      const data = await response.json();
      setHistory(data.photos);
    } catch (err) {
      console.error("Failed to fetch history", err);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleScan = async (mock = false) => {
    setIsScanning(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/scan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mock: mock, mock_source: mock ? "test_scanner_bed.png" : "" })
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Scan failed");
      }

      const data: ScanResult = await response.json();
      setCurrentScan(data);
      fetchHistory();
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

  const handleRefineSave = async (points: number[][]) => {
    if (!currentScan || refinePhotoIndex === -1) return;

    try {
      const response = await fetch(`${API_BASE}/refine`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scan_path: currentScan.scan_path,
          photo_index: refinePhotoIndex,
          points: points
        })
      });

      if (!response.ok) {
        throw new Error("Failed to refine photo");
      }

      const data = await response.json();

      // Update the local state with the new photo URL to reflect the change immediately
      const newPhotoUrl = `${data.photo_url}?t=${Date.now()}`;

      const updatedPhotos = [...currentScan.photos];
      updatedPhotos[refinePhotoIndex] = newPhotoUrl;

      setCurrentScan({
        ...currentScan,
        photos: updatedPhotos
      });

      setIsRefineOpen(false);
      fetchHistory();

    } catch (err: any) {
      alert(`Error refining photo: ${err.message}`);
    }
  };

  const handleRotate = async (index: number, angle: number) => {
    if (!currentScan) return;
    const photoUrl = currentScan.photos[index];

    // If photoUrl contains query param, we should strip it for the key
    const cleanUrl = photoUrl.split('?')[0];

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
      const newPhotoUrl = `${cleanUrl}?t=${Date.now()}`;
      const updatedPhotos = [...currentScan.photos];
      updatedPhotos[index] = newPhotoUrl;

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
          onSave={handleRefineSave}
          imageUrl={`${API_BASE}${currentScan.scan_path}`}
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

      {error && (
        <div className="bg-red-500/10 border border-red-500/50 text-red-400 p-4 rounded-xl mb-8">
          Error: {error}
        </div>
      )}

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
                <div key={idx} className="bg-slate-900 border border-slate-700 rounded-xl overflow-hidden shadow-lg">
                  {/* Remove fixed height and object-cover to show full image */}
                  <div className="p-4 flex justify-center bg-black/20">
                    <img
                      src={photo.startsWith("http") ? photo : `${API_BASE}${photo}`}
                      alt={`Photo ${idx}`}
                      className="max-w-full h-auto shadow-md"
                    />
                  </div>

                  <div className="p-4 flex justify-between items-center bg-slate-800 border-t border-slate-700">
                    <span className="text-sm font-medium text-slate-400">Photo {idx + 1}</span>

                    <div className="flex gap-2">
                      <button
                        onClick={() => handleRotate(idx, -90)}
                        className="p-2 bg-slate-700 text-slate-300 rounded hover:bg-slate-600 hover:text-white transition-colors"
                        title="Rotate Left"
                      >
                        ↺
                      </button>
                      <button
                        onClick={() => handleRotate(idx, 90)}
                        className="p-2 bg-slate-700 text-slate-300 rounded hover:bg-slate-600 hover:text-white transition-colors"
                        title="Rotate Right"
                      >
                        ↻
                      </button>
                      <div className="w-px h-6 bg-slate-600 mx-2 self-center"></div>
                      <button
                        onClick={() => openRefine(idx)}
                        className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-500 transition-colors text-sm font-semibold"
                      >
                        Refine Crop
                      </button>
                    </div>
                  </div>
                </div>
              ))}
              {!currentScan && (
                <p className="text-slate-500 italic text-center py-8">Photos will appear here after scanning</p>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
