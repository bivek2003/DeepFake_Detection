import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Shield, ArrowRight } from 'lucide-react';
import IntroAnimation from '../components/IntroAnimation';
import MethodologyModal from '../components/MethodologyModal';

export default function LandingPage() {
    const navigate = useNavigate();
    const [showIntro, setShowIntro] = useState(true);
    const [showMethodology, setShowMethodology] = useState(false);

    return (
        <div className="min-h-screen bg-gray-900 text-white overflow-hidden">
            {showIntro && <IntroAnimation onComplete={() => setShowIntro(false)} />}
            <MethodologyModal isOpen={showMethodology} onClose={() => setShowMethodology(false)} />

            {/* Background Gradients */}
            <div className="absolute top-0 left-0 w-full h-full overflow-hidden z-0 pointer-events-none">
                <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] bg-primary-600 rounded-full blur-[120px] opacity-20" />
                <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] bg-purple-600 rounded-full blur-[120px] opacity-20" />
            </div>

            <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-screen flex flex-col justify-center">

                {/* Hero Section */}
                <div className="grid lg:grid-cols-2 gap-12 items-center">
                    <div className="space-y-8 animate-fade-in-up">
                        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary-500/10 border border-primary-500/30 text-primary-300 text-sm font-medium">
                            <Shield className="w-4 h-4" />
                            <span>Production Grade Security</span>
                        </div>

                        <h1 className="text-5xl lg:text-7xl font-bold font-display leading-tight">
                            Detect <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary-400 to-purple-400">Deepfakes</span> with Precision
                        </h1>

                        <p className="text-xl text-gray-400 max-w-lg">
                            Advanced AI forensics to verify the authenticity of digital media. Protect truth in the age of generative AI.
                        </p>

                        <div className="flex flex-col sm:flex-row gap-4">
                            <button
                                onClick={() => navigate('/dashboard')}
                                className="group px-8 py-4 bg-primary-600 hover:bg-primary-500 text-white rounded-xl font-semibold transition-all shadow-[0_0_20px_rgba(79,70,229,0.3)] hover:shadow-[0_0_30px_rgba(79,70,229,0.5)] flex items-center justify-center gap-2"
                            >
                                Launch Dashboard
                                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                            </button>

                            <button
                                onClick={() => setShowMethodology(true)}
                                className="px-8 py-4 bg-white/5 hover:bg-white/10 text-white rounded-xl font-semibold transition-all border border-white/10 backdrop-blur-sm"
                            >
                                Learn Methodology
                            </button>
                        </div>

                        <div className="grid grid-cols-3 gap-8 pt-8 border-t border-white/10">
                            <div>
                                <h3 className="text-3xl font-bold text-white">99.4%</h3>
                                <p className="text-gray-500 text-sm">Detection Accuracy</p>
                            </div>
                            <div>
                                <h3 className="text-3xl font-bold text-white">&lt;100ms</h3>
                                <p className="text-gray-500 text-sm">Processing Time</p>
                            </div>
                            <div>
                                <h3 className="text-3xl font-bold text-white">24/7</h3>
                                <p className="text-gray-500 text-sm">Real-time Analysis</p>
                            </div>
                        </div>
                    </div>

                    {/* Visual Showcase */}
                    <div className="relative hidden lg:block animate-float">
                        <div className="relative z-10 bg-gray-800/50 backdrop-blur-xl border border-white/10 rounded-2xl p-6 shadow-2xl">
                            <div className="aspect-video bg-gray-900 rounded-lg overflow-hidden relative">
                                <div className="absolute inset-0 flex items-center justify-center">
                                    <div className="w-full h-full bg-gradient-to-br from-primary-500/20 to-purple-500/20" />
                                    <div className="absolute grid grid-cols-8 grid-rows-8 w-full h-full opacity-20">
                                        {Array.from({ length: 64 }).map((_, i) => (
                                            <div key={i} className="border border-primary-500/30" />
                                        ))}
                                    </div>
                                </div>

                                {/* Simulated Analysis UI */}
                                <div className="absolute top-4 right-4 bg-black/60 backdrop-blur-md px-3 py-1 rounded-full text-xs text-success-400 border border-success-500/30 flex items-center gap-2">
                                    <div className="w-2 h-2 rounded-full bg-success-400 animate-pulse" />
                                    CONFIRMED REAL
                                </div>

                                <div className="absolute bottom-4 left-4 right-4 bg-black/60 backdrop-blur-md p-4 rounded-xl border border-white/10">
                                    <div className="flex justify-between items-end mb-2">
                                        <span className="text-gray-400 text-xs">CONFIDENCE SCORE</span>
                                        <span className="text-white font-mono font-bold">99.98%</span>
                                    </div>
                                    <div className="h-1 bg-gray-700 rounded-full overflow-hidden">
                                        <div className="h-full bg-success-500 w-[99.98%]" />
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Decor elements */}
                        <div className="absolute -top-10 -right-10 w-24 h-24 bg-primary-500 rounded-full blur-[50px] opacity-40" />
                        <div className="absolute -bottom-10 -left-10 w-32 h-32 bg-purple-500 rounded-full blur-[60px] opacity-40" />
                    </div>
                </div>
            </div>
        </div>
    );
}
