import React, { useEffect, useState } from 'react';
import { Shield, Scan } from 'lucide-react';

interface IntroAnimationProps {
    onComplete: () => void;
}

export default function IntroAnimation({ onComplete }: IntroAnimationProps) {
    const [stage, setStage] = useState(0);

    useEffect(() => {
        // Stage 0: Initial Scan (0ms)
        // Stage 1: Analyzing grid (1500ms)
        // Stage 2: Verified (3000ms)
        // Stage 3: Fade out (4000ms)

        const t1 = setTimeout(() => setStage(1), 1000);
        const t2 = setTimeout(() => setStage(2), 2500);
        const t3 = setTimeout(() => setStage(3), 3500);
        const t4 = setTimeout(onComplete, 4000);

        return () => {
            clearTimeout(t1);
            clearTimeout(t2);
            clearTimeout(t3);
            clearTimeout(t4);
        };
    }, [onComplete]);

    if (stage === 4) return null;

    return (
        <div className={`fixed inset-0 z-50 bg-black flex items-center justify-center transition-opacity duration-700 ${stage === 3 ? 'opacity-0 pointer-events-none' : 'opacity-100'}`}>

            {/* Background Matrix Effect (Simplified) */}
            <div className="absolute inset-0 overflow-hidden opacity-20">
                <div className="grid grid-cols-12 gap-1 h-full animate-pulse-slow">
                    {Array.from({ length: 48 }).map((_, i) => (
                        <div key={i} className="text-primary-500 text-xs font-mono opacity-50 flex flex-col">
                            {Array.from({ length: 10 }).map((_, j) => (
                                <span key={j}>{Math.random() > 0.5 ? '1' : '0'}</span>
                            ))}
                        </div>
                    ))}
                </div>
            </div>

            <div className="relative text-center">
                {/* Central Icon */}
                <div className="relative w-32 h-32 mx-auto mb-8">
                    <div className={`absolute inset-0 border-4 border-primary-500 rounded-full animate-ping ${stage >= 2 ? 'opacity-0' : 'opacity-30'}`} />
                    <div className="absolute inset-0 flex items-center justify-center bg-gray-900 rounded-full border-2 border-primary-600 shadow-[0_0_50px_rgba(79,70,229,0.5)]">
                        {stage < 2 ? (
                            <Scan className="w-16 h-16 text-primary-400 animate-pulse" />
                        ) : (
                            <Shield className="w-16 h-16 text-success-400 animate-bounce-in" />
                        )}
                    </div>

                    {/* Scanning Line */}
                    {stage < 2 && (
                        <div className="absolute top-0 left-0 w-full h-1 bg-primary-400 shadow-[0_0_15px_rgba(99,102,241,1)] animate-scan" />
                    )}
                </div>

                {/* Text Status */}
                <div className="space-y-2">
                    <h2 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary-400 to-purple-400 font-display tracking-wider">
                        DEEPFAKE DETECTION
                    </h2>
                    <div className="h-6">
                        <p className="text-primary-300 font-mono text-sm tracking-widest uppercase">
                            {stage === 0 && "Initializing Core..."}
                            {stage === 1 && "Analyzing Patterns..."}
                            {stage === 2 && "System Verified."}
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
