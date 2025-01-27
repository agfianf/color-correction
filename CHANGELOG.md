# Changelog

## [v0.0.1a2] - 2025-01-27

### 🚀 New Features
- **feat:** add GitHub Actions workflow for publishing package to PyPI and update README with installation and usage instructions (3d07d2c)
- **feat:** update versioning scheme and enhance project metadata for clarity (6f0fab4)
- **feat:** update model folder path in downloader utility for improved file management (b8bf5d9)
- **feat:** initialize color correction module and update project metadata (c42ca92)
- **feat(dependencies):** add shapely and colour-science dependencies for enhanced image processing (15cb63b)
- **feat:** add image and geometry processing utilities for patch extraction and analysis (77769ed)
- **feat:** add color checker reference and enhance YOLOv8 detection with patch extraction (2458ce5)
- **feat:** implement base class and least squares regression for image correction (f2f8443)
- **feat(core/card_detection/yolov8):** add auto download model onnx based on spec - add device specifications schema and detection utilities (954d631)
- **feat(build):** add Makefile target for exporting YOLO model to ONNX format (b8b86bf)

### 🛠️ Improvements
- **refactor:** remove debug print statement from nms function (b369046)
- **refactor:** YOLOv8CardDetector class to improve documentation and add half-precision support; adjust font size in draw_detections function (10fd6c2)

### 🐛 Bug Fixes
- **fix(core):** fixing drop model performance by: - Update YOLOv8CardDetector to enhance input preparation and adjust IoU threshold; - improve image scaling and tensor conversion (9bd9fd9)

### 📚 Documentation
- **docs(yolo_utils):** enhance NMS function documentation for clarity and detail (c23287c)
- **docs(README):** update links and remove outdated content (5c58cc3)
- **docs(yolo_utils):** enhance function documentation for clarity and completeness (863c459)

### 🧹 Chores
- **chore:** update .gitignore to exclude pytest and ruff cache directories (4584073)
- **chore:** update .gitignore to exclude coverage files (1fa5c9d)
- **chore(deps):** update dependencies and add new packages (80b9e22)

### 🧪 Tests
- **test:** add return type annotation to test_detector_init function (0fdd5c4)
- **test:** add unit tests for YOLOv8 detector and NMS functions (e92ad54)

### 📦 Build
- **build:** update dependencies and enhance testing workflow with coverage (e45a9f2)
- **build:** add test command to Makefile for running pytest (b958500)

### ⚙️ CI
- **ci:** remove push trigger from tests workflow (4f0f9e9)
- **ci:** update workflow to use ruff for linting and formatting checks (cfdd7cd)
- **ci:** enhance GitHub Actions workflow with caching and pre-commit checks (e8fa935)
- **ci:** add GitHub Actions workflow for automated testing (70f649c)

### 🔄 Merges
- **Merge pull request #2 from agfianf/feat/add-least-squares-correction** (d69c03e)
- **Merge pull request #1 from agfianf/feat/add-yolov8-detector** (3bb33f9)

### 📝 Initial Setup
- **Initialize project with Python version, .gitignore, VSCode settings, pre-commit configuration, and pyproject.toml** (71a8c74)
- **Add README.md for Color Correction package documentation** (2b35650)