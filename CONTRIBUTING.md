# Contributing to Deepfake Detection System

Thank you for your interest in contributing to our deepfake detection system! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues
1. Check existing issues to avoid duplicates
2. Use the issue template when available
3. Provide detailed information including:
   - Steps to reproduce
   - Expected vs actual behavior
   - System information
   - Error logs/screenshots

### Submitting Pull Requests
1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes following our coding standards
4. Add tests for new functionality
5. Update documentation as needed
6. Submit a pull request

## 🔧 Development Setup

### Prerequisites
- Python 3.8+
- Git
- CUDA-capable GPU (optional but recommended)

### Local Setup
```bash
# Clone your fork
git clone https://github.com/yourusername/deepfake-detector.git
cd deepfake-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
make install-dev

# Run tests to verify setup
make test
```

## 📝 Coding Standards

### Python Style
- Follow PEP 8 guidelines
- Use Black for code formatting: `make format`
- Use type hints where possible
- Maximum line length: 88 characters (Black default)

### Code Quality
- Write docstrings for all public functions/classes
- Add type annotations
- Include unit tests for new features
- Maintain test coverage >90%
- Use meaningful variable and function names

### Example Function Documentation
```python
def extract_faces_from_video(self, video_path: Union[str, Path], 
                           return_metadata: bool = False) -> Union[List[np.ndarray], 
                                                                  Tuple[List[np.ndarray], Dict]]:
    """
    Extract face crops from video
    
    Args:
        video_path: Path to video file
        return_metadata: Whether to return processing metadata
        
    Returns:
        List of face crops or tuple with crops and metadata
        
    Raises:
        ValueError: If video file cannot be opened
        
    Example:
        >>> processor = VideoProcessor()
        >>> faces = processor.extract_faces_from_video("video.mp4")
        >>> print(f"Extracted {len(faces)} faces")
    """
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_video_processor.py -v

# Run with coverage
pytest --cov=src/deepfake_detector --cov-report=html
```

### Writing Tests
- Place tests in the `tests/` directory
- Mirror the source structure (e.g., `tests/data/test_video_processor.py`)
- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies

### Test Example
```python
import pytest
from unittest.mock import Mock, patch
from deepfake_detector.data import VideoProcessor

class TestVideoProcessor:
    def test_extract_faces_success(self):
        processor = VideoProcessor()
        # Test implementation
        
    def test_extract_faces_no_video_file(self):
        processor = VideoProcessor()
        with pytest.raises(ValueError, match="Could not open"):
            processor.extract_faces_from_video("nonexistent.mp4")
```

## 📚 Documentation

### Requirements
- Update README.md for major features
- Add docstrings to all public APIs
- Include usage examples
- Update CHANGELOG.md

### Documentation Style
- Use clear, concise language
- Include code examples
- Explain the "why" not just the "what"
- Keep documentation up-to-date with code changes

## 🗂️ Project Structure

```
deepfake-detector/
├── src/deepfake_detector/          # Main package
│   ├── data/                       # Data processing
│   ├── models/                     # Model architectures (Phase 2+)
│   ├── utils/                      # Utilities
│   └── __init__.py
├── tests/                          # Test files
├── examples/                       # Usage examples
├── docs/                          # Additional documentation
└── scripts/                       # Utility scripts
```

## 🎯 Contribution Areas

### Phase 1 (Current) - Data Pipeline
- [ ] Additional dataset integrators
- [ ] More audio feature extraction methods
- [ ] Video quality assessment improvements
- [ ] Performance optimizations
- [ ] Bug fixes and edge case handling

### Future Phases
- [ ] Model implementations (Phase 2)
- [ ] API development (Phase 3)
- [ ] Frontend interfaces (Phase 4)
- [ ] Deployment tools (Phase 5)
- [ ] Performance optimizations (Phase 6)
- [ ] Ethical AI features (Phase 7)

## 🏷️ Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or improvement
- `documentation`: Documentation needs
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed
- `phase-1`, `phase-2`, etc.: Specific to project phases

## 🔄 Release Process

### Version Numbers
We use semantic versioning (SemVer): `MAJOR.MINOR.PATCH`
- MAJOR: Incompatible API changes
- MINOR: New functionality (backward compatible)
- PATCH: Bug fixes (backward compatible)

### Release Checklist
- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in appropriate files
- [ ] Tagged release in Git

## 📞 Getting Help

### Communication Channels
- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: General questions and ideas
- Email: your.email@example.com (maintainer)

### Response Times
- Issues: We aim to respond within 48 hours
- Pull requests: We aim to review within 1 week
- Security issues: Contact maintainer directly

## 🏆 Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes
- Special recognition for significant contributions

## 📋 Code Review Process

### For Contributors
- Self-review your code before submitting
- Ensure tests pass locally
- Write clear commit messages
- Respond to feedback promptly

### For Reviewers
- Be constructive and respectful
- Focus on code quality and maintainability
- Test the changes when possible
- Approve when ready or request changes with clear feedback

## 🔒 Security

### Reporting Security Issues
- Do NOT open a public issue
- Email security concerns to: security@yourproject.com
- We will respond within 24 hours
- We will coordinate disclosure responsibly

### Security Best Practices
- Never commit API keys, passwords, or secrets
- Validate all inputs
- Use secure coding practices
- Keep dependencies updated

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the Deepfake Detection System! 🎭
