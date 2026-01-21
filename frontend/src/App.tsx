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

      <main className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left: Previous Scan Preview */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-slate-800 p-6 rounded-2xl border border-slate-700 h-fit">
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
            <div className="grid grid-cols-1 gap-4">
              {currentScan?.photos.map((photo, idx) => (
                <div key={idx} className="bg-slate-900 border border-slate-700 rounded-xl overflow-hidden shadow-lg group relative">
                  <img src={photo.startsWith("http") ? photo : `${API_BASE}${photo}`} alt={`Photo ${idx}`} className="w-full h-48 object-cover group-hover:scale-105 transition-transform duration-500" />
                  <div className="p-4 flex justify-between items-center bg-slate-800/80 backdrop-blur-sm absolute bottom-0 w-full translate-y-full group-hover:translate-y-0 transition-transform">
                    <span className="text-sm font-medium">Photo {idx + 1}</span>
                    <button
                      onClick={() => openRefine(idx)}
                      className="text-xs px-3 py-1 bg-blue-600 rounded-md hover:bg-blue-500 transition-colors"
                    >
                      Refine
                    </button>
                  </div>
                </div>
              ))}
              {!currentScan && history.length > 0 && (
                <p className="text-slate-500 italic text-center py-8">Load a scan to see isolated photos</p>
              )}
              {!currentScan && history.length === 0 && (
                <p className="text-slate-500 italic text-center py-8">Photos will appear here after scanning</p>
              )}
            </div>
          </div>

          {/* History Sidebar */}
          <div className="bg-slate-800 p-6 rounded-2xl border border-slate-700">
            <h2 className="text-xl font-bold mb-4 text-slate-400">Library</h2>
            <div className="grid grid-cols-3 gap-2">
              {history.slice(0, 9).map((photo, idx) => (
                <div key={idx} className="aspect-square bg-slate-900 rounded-lg overflow-hidden border border-slate-700 opacity-60 hover:opacity-100 transition-opacity cursor-pointer">
                  <img src={`${API_BASE}${photo}`} alt="History" className="w-full h-full object-cover" />
                </div>
              ))}
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
