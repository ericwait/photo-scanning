import { useState, useRef, useEffect } from 'react';

interface RefineModalProps {
    isOpen: boolean;
    onClose: () => void;
    onSave: (points: number[][]) => void;
    imageUrl: string;
}

export function RefineModal({ isOpen, onClose, onSave, imageUrl }: RefineModalProps) {
    const [points, setPoints] = useState<number[][]>([]);
    const imgRef = useRef<HTMLImageElement>(null);

    useEffect(() => {
        if (!isOpen) setPoints([]);
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
            <div className="bg-slate-800 p-6 rounded-2xl w-full max-w-5xl h-[90vh] flex flex-col relative shadow-2xl border border-slate-700">
                <div className="flex justify-between items-center mb-4">
                    <h3 className="text-2xl font-bold">Refine Crop</h3>
                    <div className="flex gap-2 text-sm text-slate-400">
                        <span>Click the 4 corners of the photo in any order.</span>
                        <span className="text-emerald-400 font-bold">{points.length} / 4 selected</span>
                    </div>
                </div>


                {/* 
                    Let's restart the render part for the image container to be robust 
                */}
                <div className="flex-1 overflow-hidden flex items-center justify-center">
                    <div className="relative inline-block">
                        <img
                            ref={imgRef}
                            src={imageUrl}
                            className="max-h-[70vh] max-w-full object-contain block cursor-crosshair"
                            onClick={handleImageClick}
                        />
                        {points.map((p, i) => {
                            if (!imgRef.current) return null;
                            // We need to map natural coords (p) back to displayed coords
                            // This requires a re-render when image loads/resizes, but for now simple math on render
                            // Wait, we can't get rect during render efficiently without specific state.
                            // BUT, if we position markers using percentages, it works!
                            // p[0] is x, p[1] is y. 
                            const naturalW = imgRef.current.naturalWidth;
                            const naturalH = imgRef.current.naturalHeight;

                            if (!naturalW) return null; // Image not loaded yet

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

                        {/* We need to set viewBox on the SVG to match the image natural size for the polygon points to align */}
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
                        onClick={() => points.length === 4 && onSave(points)}
                        disabled={points.length !== 4}
                        className="px-6 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg font-bold transition-colors"
                    >
                        Save Crop
                    </button>
                </div>
            </div>
        </div>
    );
}
