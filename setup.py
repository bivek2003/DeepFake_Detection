"""
Setup configuration for Deepfake Detection App
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="deepfake-detector",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive deepfake detection system for images, videos, and audio",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/deepfake-detector",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
        "Topic :: Multimedia :: Sound/Audio",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "jupyter>=1.0.0",
        ],
        "gpu": [
            "torch[cu117]>=1.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "deepfake-detector=deepfake_detector.cli:main",
        ],
    },
)
