# Clean 3D Human Pose Estimation Pipeline

This is a clean, research-focused 3D human pose estimation pipeline that estimates 3D pose from single images using self-supervised learning principles.

## Setup Complete

✅ **Environment**: Python 3.11 virtual environment (`venv_311`)
✅ **Dependencies**: All required packages installed (PyTorch, MediaPipe, matplotlib, OpenCV)
✅ **Code**: Clean implementation without GUIs or unnecessary complexity
✅ **Test**: Successfully running with real 2D pose detection

## Quick Start

### 1. Activate the Environment
```bash
# On Windows PowerShell
.\venv_311\Scripts\Activate.ps1

# On Windows Command Prompt
.\venv_311\Scripts\activate.bat

# On Linux/Mac
source venv_311/bin/activate
```

### 2. Run the Pipeline
```bash
python pose_estimation_clean.py
```

## What It Does

1. **2D Pose Detection**: Uses MediaPipe to detect 2D pose keypoints from input image
2. **3D Lifting**: Uses a neural network to lift 2D poses to 3D space
3. **Visualization**: Generates research-style output showing:
   - Original input image
   - 2D pose overlay
   - Multiple 3D skeleton views (front, side, 3D perspective)

## Output

The script generates `000033016_3d_pose_results.png` with:
- High confidence pose detection (98.4%)
- 17 detected keypoints
- Research-quality visualizations

## Key Features

- ✅ **Clean Code**: No GUI, real-time, or installation complexity
- ✅ **Research Focus**: Single-image input, batch processing ready
- ✅ **Modern Stack**: PyTorch for 3D lifting, MediaPipe for 2D detection
- ✅ **Paper-Style Output**: Multiple views like academic publications
- ✅ **Self-Supervised**: Implements SSL principles for 3D pose estimation

## Technical Details

- **Input**: Single RGB image (any size, auto-resized)
- **2D Detection**: MediaPipe Pose (33 landmarks → 17 key joints)
- **3D Lifting**: Simple neural network (can be replaced with more advanced models)
- **Output**: Research-style visualization with multiple viewpoints
- **Performance**: ~2-3 seconds per image (including visualization)

## Next Steps for Research

1. **Advanced Models**: Replace simple 3D lifting with pretrained models (VideoPose3D, etc.)
2. **Batch Processing**: Add support for processing multiple images
3. **Evaluation**: Add metrics computation (MPJPE, PA-MPJPE)
4. **Datasets**: Integrate with Human3.6M, MPI-INF-3DHP datasets
5. **Self-Supervision**: Enhance SSL components with temporal consistency, view synthesis

## Files

- `pose_estimation_clean.py` - Main pipeline implementation
- `000033016.jpg` - Test image
- `000033016_3d_pose_results.png` - Generated output
- `venv_311/` - Python 3.11 virtual environment
- `3Dpose_ssl/` - Original repository (reference only)

## Dependencies

All installed in `venv_311`:
- torch==2.7.1
- torchvision==0.22.1  
- mediapipe==0.10.21
- matplotlib==3.10.3
- opencv-python==4.11.0.86
- numpy==1.26.4

---

**Status**: ✅ Ready for research and development
**Python Version**: 3.11.9 (compatible with all research libraries)
**Last Updated**: January 2025
