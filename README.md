# 3D Human Pose Estimation - Clean & Interactive

A clean, research-style implementation of 3D human pose estimation with interactive visualization and correction tools.

## ✅ Problem Solved: "2D poses look good but 3D poses look weird"

This project provides a complete solution for single-image 3D human pose estimation with interactive tools to fix and improve the results.

## 🎯 Key Features

- **MediaPipe 2D Detection**: High-quality 2D pose detection
- **Neural 3D Lifting**: Convert 2D poses to 3D with anatomical constraints
- **Interactive Adjustment**: Real-time 3D pose correction with sliders
- **Multiple Viewing Modes**: Front, side, and 3/4 views
- **Anatomical Constraints**: Realistic bone lengths and joint angles

## 🚀 Quick Start

### 1. Main Pipeline
```bash
python pose_estimation_clean.py
```
Processes images and generates 3D poses using the core pipeline.

### 2. Interactive Pose Adjuster ⭐ (Recommended)
```bash
python pose_adjuster.py
```
Opens an interactive GUI to adjust and improve 3D poses in real-time:
- 🖱️ Mouse rotation of 3D view
- 🎚️ Sliders for depth, height, body part positioning
- ⌨️ Keyboard shortcuts (r=reset, s=save, c=constraints)
- 🔧 Anatomical constraint enforcement

## 📁 Project Structure

```
📦 3D Human Pose Estimation/
├── 🎮 pose_adjuster.py              # Interactive adjustment tool ⭐
├── 📄 pose_estimation_clean.py      # Main pose estimation pipeline  
├── 📸 Test images (.jpg)            # Sample input images
├── 🖼️  Result visualizations (.png) # Generated comparisons
├── 📚 Documentation (.md)           # Setup and usage guides
└── 🐍 venv_311/                     # Python 3.11 environment
```

## 🔧 Installation

1. **Python Environment** (Python 3.11 recommended)
```bash
# Create virtual environment
python -m venv venv_311
# Activate (Windows)
venv_311\Scripts\activate
```

2. **Install Dependencies**
```bash
pip install torch torchvision matplotlib opencv-python mediapipe numpy
```

## 🎯 Key Improvements

### ✅ Fixed Issues:
- **Random depth assignment** → Anatomically-based depth estimation
- **Poor coordinate transformation** → Proper 3D projection
- **No anatomical constraints** → Bone length and angle limits
- **Floating poses** → Ground contact enforcement
- **Asymmetric poses** → Left-right symmetry

### 📊 Results:
- `*_enhanced_3d_comparison.png` - Method comparisons
- `*_FIXED_comparison.png` - Before/after improvements

## 🎮 Interactive Controls

### Pose Adjuster (`pose_adjuster.py`):
- **Depth Scale**: Overall depth scaling
- **Height Offset**: Vertical positioning
- **Arm Depth**: Forward/backward arm positioning  
- **Leg Depth**: Leg depth adjustment
- **Torso Lean**: Forward/backward torso lean
- **Head Depth**: Head positioning

### Keyboard Shortcuts:
- `r` - Reset pose
- `s` - Save pose
- `c` - Apply constraints
- `1` - Front view
- `2` - Side view  
- `3` - 3/4 view

## 📈 Usage Examples

### Process Single Image:
```python
# Load and process
pose_3d = estimator.estimate_3d_pose("image.jpg")

# Visualize
estimator.visualize_results(save_path="result.png")
```

### Interactive Adjustment:
```python
# Load the interactive adjuster
python pose_adjuster.py

# Use sliders to improve the pose
# Save when satisfied with results
```

## 🎯 Success Metrics

- ✅ **2D Detection**: High-confidence MediaPipe detection
- ✅ **3D Realism**: Anatomically plausible poses
- ✅ **Interactive**: Real-time adjustment capability
- ✅ **Clean Code**: Research-ready implementation

## 📚 Documentation

- `3D_POSE_FIX_GUIDE.md` - Detailed usage guide
- `SETUP_GUIDE.md` - Installation instructions

---

**Result**: Transform weird-looking 3D poses into natural, anatomically correct human poses with interactive fine-tuning capability!
