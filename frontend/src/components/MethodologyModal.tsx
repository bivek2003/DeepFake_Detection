import React from 'react';
import { X, Server, Shield, Cpu } from 'lucide-react';

interface MethodologyModalProps {
    isOpen: boolean;
    onClose: () => void;
}

export default function MethodologyModal({ isOpen, onClose }: MethodologyModalProps) {
    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-[60] flex items-center justify-center p-4">
            {/* Backdrop */}
            <div
                className="absolute inset-0 bg-black/80 backdrop-blur-sm animate-fade-in"
                onClick={onClose}
            />

            {/* Modal Content */}
            <div className="relative bg-gray-900 border border-white/10 rounded-2xl w-full max-w-3xl max-h-[90vh] overflow-y-auto shadow-2xl animate-bounce-in">
                <div className="sticky top-0 bg-gray-900/95 backdrop-blur border-b border-white/10 p-6 flex items-center justify-between z-10">
                    <h2 className="text-2xl font-bold text-white font-display">Detection Methodology</h2>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-white/10 rounded-lg transition-colors text-gray-400 hover:text-white"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                <div className="p-6 space-y-8">
                    {/* Section 1: Architecture */}
                    <section className="space-y-4">
                        <div className="flex items-center gap-3 text-primary-400">
                            <Cpu className="w-6 h-6" />
                            <h3 className="text-xl font-semibold">Model Architecture</h3>
                        </div>
                        <div className="bg-gray-800/50 rounded-xl p-5 border border-white/5 space-y-3">
                            <p className="text-gray-300 leading-relaxed">
                                Our system utilizes an <strong>EfficientNet-B4</strong> backbone, a state-of-the-art Convolutional Neural Network (CNN) optimized for image classification. This backbone extracts high-level features from input frames, which are then processed by a custom <strong>3-layer classification head</strong> (1024 → 512 → 1) to determine authenticity.
                            </p>
                            <ul className="list-disc list-inside text-gray-400 text-sm space-y-1 ml-2">
                                <li>Pre-trained on ImageNet for robust feature extraction</li>
                                <li>Fine-tuned on Celeb-DF and FaceForensics++ datasets</li>
                                <li>Uses Global Average Pooling to handle varying input sizes</li>
                            </ul>
                        </div>
                    </section>

                    {/* Section 2: Analysis Pipeline */}
                    <section className="space-y-4">
                        <div className="flex items-center gap-3 text-purple-400">
                            <Server className="w-6 h-6" />
                            <h3 className="text-xl font-semibold">Analysis Pipeline</h3>
                        </div>
                        <div className="grid md:grid-cols-2 gap-4">
                            <div className="bg-gray-800/50 rounded-xl p-5 border border-white/5">
                                <h4 className="text-white font-medium mb-2">Image Analysis</h4>
                                <p className="text-gray-400 text-sm">
                                    Static images undergo face detection using MTCNN. Cropped faces are normalized and passed through the model. Gradient-weighted Class Activation Mapping (Grad-CAM) is generated to visualize manipulated regions.
                                </p>
                            </div>
                            <div className="bg-gray-800/50 rounded-xl p-5 border border-white/5">
                                <h4 className="text-white font-medium mb-2">Video Analysis</h4>
                                <p className="text-gray-400 text-sm">
                                    Videos are processed frame-by-frame. Temporal consistency is analyzed by aggregating frame-level predictions using a smoothing algorithm to prevent flickering verdicts.
                                </p>
                            </div>
                        </div>
                    </section>

                    {/* Section 3: Security & Privacy */}
                    <section className="space-y-4">
                        <div className="flex items-center gap-3 text-success-400">
                            <Shield className="w-6 h-6" />
                            <h3 className="text-xl font-semibold">Security & Privacy</h3>
                        </div>
                        <div className="bg-gray-800/50 rounded-xl p-5 border border-white/5">
                            <p className="text-gray-300 leading-relaxed">
                                All media is processed in a secure, isolated environment. Uploaded files are hashed (SHA-256) for integrity verification and are automatically retained for a limited period for audit purposes before deletion.
                            </p>
                        </div>
                    </section>
                </div>

                <div className="p-6 border-t border-white/10 bg-gray-900/50">
                    <button
                        onClick={onClose}
                        className="w-full py-3 bg-primary-600 hover:bg-primary-500 text-white rounded-xl font-bold transition-all"
                    >
                        Close Methodology
                    </button>
                </div>
            </div>
        </div>
    );
}
