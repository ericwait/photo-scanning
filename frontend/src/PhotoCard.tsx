
import { useState, useEffect } from 'react';

export interface PhotoData {
    url: string;
    points: number[][];
}

interface PhotoCardProps {
    photo: PhotoData;
    index: number;
    onRotate: (index: number, angle: number) => void;
    onRefineOpen: (index: number) => void;
    onUpdateSettings: (index: number, photo: PhotoData, settings: { contrast: number; autoContrast: boolean; autoWb: boolean }) => void;
    globalSettings: {
        contrast: number;
        autoContrast: boolean;
        autoWb: boolean;
    };
    scanId: string;
}

export function PhotoCard({ photo, index, onRotate, onRefineOpen, onUpdateSettings, globalSettings, scanId }: PhotoCardProps) {
    const [isSettingsOpen, setIsSettingsOpen] = useState(false);


    // Local state for overrides. Initialize with global or defaults?
    // Since we don't store per-photo settings in the backend (yet), we start with global defaults 
    // or neutral values?
    // The user wants to "overwrite". 
    // Let's initialize with the global settings passed in props, assuming this is a fresh photo.
    // If the photo has been updated, we don't persist its specific settings in the parent state currently.
    // This is a limitation: if you change settings, then close/open, they might reset if we don't persist them in App.tsx.
    // implementing full persistence in App.tsx might be too big of a refactor. 
    // For now, let's keep state local to the card.

    const [contrast, setContrast] = useState(globalSettings.contrast);
    const [autoContrast, setAutoContrast] = useState(globalSettings.autoContrast);
    const [autoWb, setAutoWb] = useState(globalSettings.autoWb);
    const [isProcessing, setIsProcessing] = useState(false);

    // Reset local overrides when a new scan (page) is loaded
    useEffect(() => {
        setContrast(globalSettings.contrast);
        setAutoContrast(globalSettings.autoContrast);
        setAutoWb(globalSettings.autoWb);
        setIsSettingsOpen(false); // Optionally close the panel on new scan
    }, [scanId]);

    const handleApply = async () => {
        setIsProcessing(true);
        try {
            await onUpdateSettings(index, photo, { contrast, autoContrast, autoWb });
        } finally {
            setIsProcessing(false);
        }
    };

    const displayUrl = photo.url.startsWith("http") ? photo.url : `http://localhost:8001${photo.url}`;

    return (
        <div className="bg-slate-900 border border-slate-700 rounded-xl overflow-hidden shadow-lg flex flex-col">
            <div className="p-4 flex-1 flex justify-center bg-black/20 relative group">
                <img
                    src={displayUrl}
                    alt={`Photo ${index + 1}`}
                    className={`max-w-full h-auto shadow-md transition-opacity ${isProcessing ? 'opacity-50' : 'opacity-100'}`}
                />
                {isProcessing && (
                    <div className="absolute inset-0 flex items-center justify-center">
                        <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                    </div>
                )}
            </div>

            <div className="bg-slate-800 border-t border-slate-700">
                <div className="p-4 flex justify-between items-center">
                    <span className="text-sm font-medium text-slate-400">Photo {index + 1}</span>

                    <div className="flex gap-2">
                        <button
                            onClick={() => setIsSettingsOpen(!isSettingsOpen)}
                            className={`p-2 rounded transition-colors ${isSettingsOpen ? 'bg-indigo-600 text-white' : 'bg-slate-700 text-slate-300 hover:bg-slate-600'}`}
                            title="Adjust Settings"
                        >
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                                <path fillRule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
                            </svg>
                        </button>
                        <div className="w-px h-6 bg-slate-600 self-center"></div>
                        <button
                            onClick={() => onRotate(index, -90)}
                            className="p-2 bg-slate-700 text-slate-300 rounded hover:bg-slate-600 hover:text-white transition-colors"
                            title="Rotate Left"
                        >
                            ↺
                        </button>
                        <button
                            onClick={() => onRotate(index, 90)}
                            className="p-2 bg-slate-700 text-slate-300 rounded hover:bg-slate-600 hover:text-white transition-colors"
                            title="Rotate Right"
                        >
                            ↻
                        </button>
                        <div className="w-px h-6 bg-slate-600 self-center"></div>
                        <button
                            onClick={() => onRefineOpen(index)}
                            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-500 transition-colors text-sm font-semibold"
                        >
                            Refine Crop
                        </button>
                    </div>
                </div>

                {isSettingsOpen && (
                    <div className="p-4 border-t border-slate-700 bg-slate-800/50 space-y-4 animate-fade-in-down">
                        <div className="flex items-center justify-between">
                            <h4 className="text-sm font-bold text-slate-300">Image Corrections</h4>
                            <button onClick={handleApply} disabled={isProcessing} className="text-xs bg-indigo-600 hover:bg-indigo-500 text-white px-3 py-1 rounded transition-colors disabled:opacity-50">
                                Apply Changes
                            </button>
                        </div>

                        {/* Auto WB */}
                        <label className="flex items-center gap-3 cursor-pointer group">
                            <div className={`w-4 h-4 rounded border flex items-center justify-center transition-colors ${autoWb ? 'bg-blue-600 border-blue-600' : 'bg-slate-900 border-slate-700'}`}>
                                {autoWb && <span className="text-white text-[10px]">✓</span>}
                            </div>
                            <input
                                type="checkbox"
                                className="hidden"
                                checked={autoWb}
                                onChange={(e) => setAutoWb(e.target.checked)}
                            />
                            <span className="text-sm text-slate-300">Auto White Balance</span>
                        </label>

                        {/* Auto Contrast */}
                        <label className="flex items-center gap-3 cursor-pointer group">
                            <div className={`w-4 h-4 rounded border flex items-center justify-center transition-colors ${autoContrast ? 'bg-indigo-600 border-indigo-600' : 'bg-slate-900 border-slate-700'}`}>
                                {autoContrast && <span className="text-white text-[10px]">✓</span>}
                            </div>
                            <input
                                type="checkbox"
                                className="hidden"
                                checked={autoContrast}
                                onChange={(e) => setAutoContrast(e.target.checked)}
                            />
                            <span className="text-sm text-slate-300">Auto Contrast</span>
                        </label>

                        {/* Manual Contrast */}
                        <div>
                            <div className="flex justify-between mb-1">
                                <label className="text-xs font-medium text-slate-400">
                                    Contrast {contrast}x
                                </label>
                            </div>
                            <input
                                type="range"
                                min="1.0"
                                max="2.0"
                                step="0.1"
                                value={contrast}
                                onChange={(e) => setContrast(Number(e.target.value))}
                                className="w-full accent-purple-500 h-1 bg-slate-900 rounded-lg appearance-none cursor-pointer"
                            />
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
