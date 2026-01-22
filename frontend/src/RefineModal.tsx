import { useState, useRef, useEffect } from 'react';

interface RefineModalProps {
    isOpen: boolean;
    onClose: () => void;
    onSave: (points: number[][], settings: { contrast: number; autoContrast: boolean; autoWb: boolean }) => void;
    imageUrl: string;
    initialSettings: {
        contrast: number;
        autoContrast: boolean;
        autoWb: boolean;
    }
}

export function RefineModal({ isOpen, onClose, onSave, imageUrl, initialSettings }: RefineModalProps) {
    const [points, setPoints] = useState<number[][]>([]);
    const [localContrast, setLocalContrast] = useState(1.0);
    const [localAutoContrast, setLocalAutoContrast] = useState(false);
    const [localAutoWb, setLocalAutoWb] = useState(false);

    const imgRef = useRef<HTMLImageElement>(null);

    useEffect(() => {
        if (isOpen) {
            setPoints([]);
            setLocalContrast(initialSettings.contrast);
            setLocalAutoContrast(initialSettings.autoContrast);
            setLocalAutoWb(initialSettings.autoWb);
        }
    }, [isOpen]);

    const handleImageClick = (e: React.MouseEvent<HTMLDivElement>) => {
        if (points.length >= 4) return;
        if (!imgRef.current) return;

        const rect = imgRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Calculate actual image coordinates scaling back up
        // We need the natural size vs display size ratio
        const scaleX = imgRef.current.naturalWidth / rect.width;
        const scaleY = imgRef.current.naturalHeight / rect.height;

        const trueX = Math.round(x * scaleX);
        const trueY = Math.round(y * scaleY);

        setPoints([...points, [trueX, trueY]]);
    };

    const handleReset = () => setPoints([]);

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-8">
            <div className="bg-slate-800 p-6 rounded-2xl w-full max-w-6xl h-[90vh] flex gap-6 relative shadow-2xl border border-slate-700">

                {/* Main Image Area */}
                <div className="flex-1 flex flex-col min-w-0">
                    <div className="flex justify-between items-center mb-4">
                        <h3 className="text-2xl font-bold">Refine Crop</h3>
                        <div className="flex gap-2 text-sm text-slate-400">
                            <span>Click the 4 corners of the photo in any order.</span>
                            <span className="text-emerald-400 font-bold">{points.length} / 4 selected</span>
                        </div>
                    </div>

                    <div className="flex-1 overflow-hidden flex items-center justify-center bg-black/20 rounded-xl border border-slate-700/50">
                        <div className="relative inline-block">
                            <img
                                ref={imgRef}
                                src={imageUrl}
                                className="max-h-[70vh] max-w-full object-contain block cursor-crosshair"
                                onClick={handleImageClick}
                            />
                            {points.map((p, i) => {
                                if (!imgRef.current) return null;
                                const naturalW = imgRef.current.naturalWidth;
                                const naturalH = imgRef.current.naturalHeight;
                                if (!naturalW) return null;

                                const leftPct = (p[0] / naturalW) * 100;
                                const topPct = (p[1] / naturalH) * 100;

                                return (
                                    <div
                                        key={i}
                                        className="absolute w-4 h-4 bg-red-500 rounded-full border-2 border-white -translate-x-1/2 -translate-y-1/2 pointer-events-none"
                                        style={{ left: `${leftPct}%`, top: `${topPct}%` }}
                                    >
                                        <span className="absolute -top-6 left-1/2 -translate-x-1/2 bg-black/50 text-white text-xs px-1 rounded">{i + 1}</span>
                                    </div>
                                )
                            })}

                            {imgRef.current?.naturalWidth && (
                                <svg
                                    className="absolute inset-0 w-full h-full pointer-events-none"
                                    viewBox={`0 0 ${imgRef.current.naturalWidth} ${imgRef.current.naturalHeight}`}
                                >
                                    <polygon
                                        points={points.map(p => `${p[0]},${p[1]}`).join(' ')}
                                        fill="rgba(52, 211, 153, 0.3)"
                                        stroke="#34d399"
                                        strokeWidth="4"
                                        vectorEffect="non-scaling-stroke"
                                    />
                                </svg>
                            )}
                        </div>
                    </div>

                    <div className="flex justify-end gap-3 mt-4">
                        <button
                            onClick={handleReset}
                            className="px-4 py-2 text-slate-300 hover:text-white hover:bg-slate-700 rounded-lg transition-colors"
                        >
                            Reset Points
                        </button>
                        <button
                            onClick={onClose}
                            className="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
                        >
                            Cancel
                        </button>
                        <button
                            onClick={() => points.length === 4 && onSave(points, {
                                contrast: localContrast,
                                autoContrast: localAutoContrast,
                                autoWb: localAutoWb
                            })}
                            disabled={points.length !== 4}
                            className="px-6 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg font-bold transition-colors"
                        >
                            Save Crop
                        </button>
                    </div>
                </div>

                {/* Sidebar: Per specific settings */}
                <div className="w-64 bg-slate-900/50 p-4 rounded-xl border border-slate-700 flex flex-col gap-6 h-full overflow-y-auto">
                    <div>
                        <h4 className="font-bold mb-1">Per-Photo Settings</h4>
                        <p className="text-xs text-slate-400">Apply corrections specifically to this crop.</p>
                    </div>

                    <div className="space-y-4">
                        {/* Auto WB */}
                        <label className="flex items-center gap-3 cursor-pointer group">
                            <div className={`w-5 h-5 rounded border flex items-center justify-center transition-colors ${localAutoWb ? 'bg-blue-600 border-blue-600' : 'bg-slate-900 border-slate-700'}`}>
                                {localAutoWb && <span className="text-white text-xs">✓</span>}
                            </div>
                            <input
                                type="checkbox"
                                className="hidden"
                                checked={localAutoWb}
                                onChange={(e) => setLocalAutoWb(e.target.checked)}
                            />
                            <span className="text-sm font-medium text-slate-300">Auto White Balance</span>
                        </label>

                        {/* Auto Contrast */}
                        <label className="flex items-center gap-3 cursor-pointer group">
                            <div className={`w-5 h-5 rounded border flex items-center justify-center transition-colors ${localAutoContrast ? 'bg-indigo-600 border-indigo-600' : 'bg-slate-900 border-slate-700'}`}>
                                {localAutoContrast && <span className="text-white text-xs">✓</span>}
                            </div>
                            <input
                                type="checkbox"
                                className="hidden"
                                checked={localAutoContrast}
                                onChange={(e) => setLocalAutoContrast(e.target.checked)}
                            />
                            <span className="text-sm font-medium text-slate-300">Auto Contrast</span>
                        </label>

                        {/* Manual Contrast */}
                        <div>
                            <div className="flex justify-between mb-2">
                                <label className="text-sm font-medium text-slate-400">
                                    Contrast Boost
                                </label>
                                <span className="text-xs text-slate-500 bg-slate-900 px-2 py-0.5 rounded">{localContrast}x</span>
                            </div>
                            <input
                                type="range"
                                min="1.0"
                                max="2.0"
                                step="0.1"
                                value={localContrast}
                                onChange={(e) => setLocalContrast(Number(e.target.value))}
                                className="w-full accent-purple-500 h-2 bg-slate-900 rounded-lg appearance-none cursor-pointer"
                            />
                        </div>
                    </div>
                </div>

            </div>
        </div>
    );
}
