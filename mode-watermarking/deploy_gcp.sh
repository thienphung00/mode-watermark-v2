#!/bin/bash
# GCP Deployment Script for Mode Watermarking

set -e

echo "🚀 Deploying Mode Watermarking to GCP..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Please run this script from the mode-watermarking root directory"
    exit 1
fi

# Clean up any existing build artifacts
echo "🧹 Cleaning up build artifacts..."
rm -rf build/ dist/ *.egg-info/
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Create a clean package
echo "📦 Creating package..."
python setup.py sdist bdist_wheel

# Show package info
echo "📋 Package created:"
ls -la dist/

# Show final directory structure
echo "📁 Final directory structure:"
find . -type f -name "*.py" | head -20

echo "✅ Mode Watermarking is ready for GCP deployment!"
echo ""
echo "Next steps:"
echo "1. Upload the dist/ directory to your GCP instance"
echo "2. Install with: pip install dist/*.whl"
echo "3. Or install directly with: pip install -r requirements.txt"
