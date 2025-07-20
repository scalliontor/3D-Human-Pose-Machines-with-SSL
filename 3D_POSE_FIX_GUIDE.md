# 3D Pose Estimation - Clean & Interactive

## Problem Solved: "The 2D seems good but the 3D looks weird" ✅

## 🎯 Core Tools (Clean Structure):

### 1. **Main Pose Estimation** (`pose_estimation_clean.py`)
**Purpose**: Core 3D human pose estimation pipeline

**Usage**:
```bash
python pose_estimation_clean.py
```

**Features**:
- MediaPipe 2D pose detection
- Neural network 3D lifting 
- Clean research-style implementation

### 2. **Interactive Pose Adjuster** (`pose_adjuster.py`) ⭐
**Purpose**: Real-time manual adjustment of 3D poses with sliders

**Usage**:
```bash
python pose_adjuster.py
```

**Features**:
- Interactive 3D visualization with mouse rotation
- Real-time sliders for adjustment:
  - Overall depth scaling
  - Height offset
  - Arm/leg positioning
  - Torso lean
  - Head position
- Keyboard shortcuts (r=reset, s=save, c=constraints)
- Multiple viewing angles
- Anatomical constraint enforcement

### 3. **Original Interactive Viewer** (`interactive_3d_viewer.py`)
**Purpose**: Advanced interactive 3D pose viewer with comprehensive controls

**Features**:
- Complex GUI with detailed controls
- Anatomical constraints system
- Multiple visualization modes

## 📁 Clean Project Structure:

```
📦 3D_Human_Pose_Machines_with_Self_supervised_Learning/
├── 📄 pose_estimation_clean.py          # Main pose estimation pipeline
├── 🎮 pose_adjuster.py                  # Interactive pose adjustment tool ⭐
├── 📸 000033016.jpg                     # Test image 1
├── 📸 000316731.jpg                     # Test image 2
├── 🖼️  *_enhanced_3d_comparison.png     # Enhanced comparison results
├── 🖼️  *_FIXED_comparison.png           # Before/after fix comparisons
├── 📚 3D_POSE_FIX_GUIDE.md             # This guide
├── 📚 SETUP_GUIDE.md                    # Setup instructions
├── 📚 README.md                         # Project README
└── 🐍 venv_311/                         # Python 3.11 virtual environment
```

## 🚀 Quick Start:

### Ready to Use - Interactive Pose Adjustment:
```bash
python pose_adjuster.py
```

This opens an interactive GUI where you can:
- 🖱️ Rotate the 3D view with your mouse
- 🎚️ Use sliders to adjust pose parameters in real-time
- ⌨️ Use keyboard shortcuts for quick actions
- 💾 Save improved poses
- 🔧 Apply anatomical constraints

## � Key Improvements Made:

### ✅ **Problem**: Random depth assignment making poses look weird
**Solution**: Anatomically-based depth estimation with logical body part relationships

### ✅ **Problem**: Poor 3D coordinate transformation  
**Solution**: Proper camera projection and aspect ratio handling

### ✅ **Problem**: No anatomical constraints
**Solution**: Bone length normalization and joint angle limits

### ✅ **Problem**: Poses floating in air
**Solution**: Ground contact enforcement

### ✅ **Problem**: Uneven body symmetry
**Solution**: Left-right symmetry constraints

## 📸 Generated Results:

- `*_enhanced_3d_comparison.png` - Shows 3 different lifting methods compared
- `*_FIXED_comparison.png` - Shows before/after direct comparison (red=old, green=fixed)

## 💡 Success: 
Your 2D poses were already good! The issue was entirely in the 3D lifting process. The interactive tools now provide natural, anatomically correct 3D poses that you can further adjust in real-time.
