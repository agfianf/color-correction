# Changelog

## [v0.0.1-rc2] - 2025-02-10
**Release Candidate with Full Feature Set and Documentation**
This release introduces several improvements:

- Updated documentation to include interactive Google Colab demos and online documentation links.
- Refactored the detection module by renaming the detection file and introducing the new `DetectionProcessor` class for clearer structure.
- Enhanced image processing utilities with improved docstrings and comments for better code readability and maintainability.

## [v0.0.1-rc1] - 2025-02-10
**Release Candidate with Full Feature Set and Documentation**
- Add docs page build with mkdocs-material. Read more [https://agfianf.github.io/color-correction/](https://agfianf.github.io/color-correction/)
- Complete docstrings for all classes and methods

## [v0.0.1b3] - 2025-02-06
**Add Analyzer Report and Bug Fixes**

### 🚀 Features
- Added comprehensive reporting functionality for color correction results
  - New `ColorCorrectionAnalyzer` class for benchmarking different correction methods
  - HTML report generation with interactive sorting and PDF export
  - Visual comparison of before/after color patches
  - Detailed ΔE metrics for patches and full images
- Enhanced image processing utilities
  - Added base64 image conversion support
  - Improved color difference calculation with rounded metrics
- Added HTML report generation templates and styling
  - Responsive design with Plus Jakarta Sans font
  - Interactive table sorting
  - PDF export functionality
  - Detailed column descriptions

### 📝 Documentation
- Added new Analyzer section in README
  - Example usage code for ColorCorrectionAnalyzer
  - Sample benchmark output visualization
- Updated version to 0.0.1b3

### 🔧 Technical Changes
- Renamed benchmark class to report for better clarity
- Added new utility modules:
  - formater.py for value formatting
  - report_generator.py for HTML generation
  - Added new constants and method definitions

## [v0.0.1b2] - 2025-02-05
Fix naming from `color-correction-asdfghjkl` to `color-correction`


## [v0.0.1b1] - 2025-02-04
**Enhanced Color Correction with Improved Documentation and Evaluation**

### ✨ Features
- Enhanced color correction with improved patch comparison and metrics
- Added polynomial correction model with configurable degrees
- Implemented comprehensive color difference evaluation

### 📚 Documentation
- Added "How it works" section with visual explanation
- Updated README with polynomial correction details
- Improved section headers for better clarity
- Added sample debug output visualization
- Enhanced usage examples with evaluation results

### 🔧 Technical
- Added `calc_color_diff_patches()` method for quality evaluation
- Implemented CIE 2000 color difference calculation
- Enhanced debug visualization capabilities
- Added support for multiple correction models


## [v0.0.1b0] - 2025-02-03

### 🔧 Improvements
- **Color Correction Core**
  - Added new correction models: polynomial, linear regression, and affine regression
  - Improved patch detection and processing pipeline
  - Added support for debug visualization outputs
  - Enhanced color patch extraction with better error handling

### 🎨 Features
- **Reference Colors**
  - Added RGB format reference colors alongside BGR
  - Improved color patch visualization and comparison tools
  - Added support for custom reference images

### 📝 Documentation
- **README Updates**
  - Simplified usage documentation with clearer examples
  - Added visual explanation of color correction workflow
  - Updated installation and usage instructions

### 🛠️ Development
- **Project Structure**
  - Reorganized core modules for better maintainability
  - Added new utility modules for image processing
  - Updated VSCode settings for better development experience

### 🔨 Build
- **Dependencies**
  - Added scikit-learn for advanced correction models
  - Updated ruff to v0.9.4
  - Added pre-commit hooks configuration

### 🧪 Testing
- **Test Coverage**
  - Added new test cases for image processing utilities
  - Improved test organization and structure



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
