#!/usr/bin/env python3
"""
Interactive 3D Pose Adjuster
Real-time adjustment of 3D poses with sliders and anatomical constraints.
This tool addresses the "weird 3D poses" issue by allowing manual correction.
"""

# 🎯 CONFIGURATION: Set your input image here
INPUT_IMAGE_PATH = "000316731.jpg"  # Change this to your desired image file
# Available options: "000033016.jpg", "000316731.jpg", or any .jpg/.png file

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.patches as mpatches
import json
import os
from pathlib import Path

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class InteractivePoseAdjuster:
    """Interactive tool to manually adjust 3D poses"""
    
    def __init__(self, pose_2d, pose_3d_initial, image=None):
        self.pose_2d = pose_2d
        self.pose_3d_original = pose_3d_initial.copy()
        self.pose_3d_current = pose_3d_initial.copy()
        self.image = image
        
        # COCO skeleton connections
        self.skeleton = [
            [1, 3], [1, 0], [2, 4], [2, 0], [0, 5], [0, 6],
            [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12],
            [11, 13], [13, 15], [12, 14], [14, 16], [11, 12]
        ]
        
        # Joint names
        self.joint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Initialize the GUI
        self.setup_gui()
    
    def setup_gui(self):
        """Set up the interactive GUI"""
        # Create figure with subplots
        self.fig = plt.figure(figsize=(18, 12))
        
        # Main 3D plot (larger)
        self.ax_3d = plt.subplot2grid((3, 4), (0, 0), colspan=3, rowspan=2, projection='3d')
        
        # 2D pose reference (smaller)
        self.ax_2d = plt.subplot2grid((3, 4), (0, 3))
        
        # Control panel area
        self.ax_controls = plt.subplot2grid((3, 4), (2, 0), colspan=4)
        self.ax_controls.axis('off')
        
        # Plot initial poses
        self.plot_2d_reference()
        self.plot_3d_pose()
        
        # Create control widgets
        self.create_controls()
        
        # Connect events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        plt.tight_layout()
        plt.show()
    
    def plot_2d_reference(self):
        """Plot 2D pose as reference"""
        self.ax_2d.clear()
        
        if self.image is not None:
            # Show original image
            image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            h, w = self.image.shape[:2]
            
            # Convert normalized coordinates to image coordinates
            pose_img = self.pose_2d.copy()
            pose_img[:, 0] *= w
            pose_img[:, 1] *= h
            
            self.ax_2d.imshow(image_rgb)
            self.ax_2d.scatter(pose_img[:, 0], pose_img[:, 1], c='red', s=30)
            
            # Plot skeleton
            for connection in self.skeleton:
                if len(connection) == 2:
                    start_idx, end_idx = connection
                    self.ax_2d.plot([pose_img[start_idx, 0], pose_img[end_idx, 0]],
                                   [pose_img[start_idx, 1], pose_img[end_idx, 1]], 
                                   'b-', linewidth=1.5, alpha=0.7)
        else:
            # Plot 2D pose without image
            self.ax_2d.scatter(self.pose_2d[:, 0], self.pose_2d[:, 1], c='red', s=50)
            
            for connection in self.skeleton:
                if len(connection) == 2:
                    start_idx, end_idx = connection
                    self.ax_2d.plot([self.pose_2d[start_idx, 0], self.pose_2d[end_idx, 0]],
                                   [self.pose_2d[start_idx, 1], self.pose_2d[end_idx, 1]], 
                                   'b-', linewidth=2, alpha=0.7)
            
            self.ax_2d.set_xlim([0, 1])
            self.ax_2d.set_ylim([0, 1])
            self.ax_2d.invert_yaxis()
        
        self.ax_2d.set_title('2D Pose Reference')
        self.ax_2d.axis('off')
    
    def plot_3d_pose(self):
        """Plot current 3D pose"""
        self.ax_3d.clear()
        
        # Plot joints with different colors for different body parts
        colors = ['red'] * 5 + ['blue'] * 6 + ['green'] * 6  # head, arms, legs
        self.ax_3d.scatter(self.pose_3d_current[:, 0], 
                          self.pose_3d_current[:, 1], 
                          self.pose_3d_current[:, 2], 
                          c=colors, s=80, alpha=0.8)
        
        # Plot skeleton connections
        for connection in self.skeleton:
            if len(connection) == 2:
                start_idx, end_idx = connection
                self.ax_3d.plot3D([self.pose_3d_current[start_idx, 0], self.pose_3d_current[end_idx, 0]],
                                 [self.pose_3d_current[start_idx, 1], self.pose_3d_current[end_idx, 1]],
                                 [self.pose_3d_current[start_idx, 2], self.pose_3d_current[end_idx, 2]], 
                                 'black', linewidth=2, alpha=0.7)
        
        # Set equal aspect ratio and limits
        max_range = 0.6
        self.ax_3d.set_xlim([-max_range, max_range])
        self.ax_3d.set_ylim([-max_range, max_range])
        self.ax_3d.set_zlim([-max_range, max_range])
        
        self.ax_3d.set_xlabel('X (Left-Right)')
        self.ax_3d.set_ylabel('Y (Down-Up)')  
        self.ax_3d.set_zlabel('Z (Back-Front)')
        
        self.ax_3d.set_title('3D Pose (Interactive)', fontsize=14, fontweight='bold')
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='red', label='Head'),
            mpatches.Patch(color='blue', label='Arms'),
            mpatches.Patch(color='green', label='Legs')
        ]
        self.ax_3d.legend(handles=legend_elements, loc='upper right')
        
        # Set initial view angle
        self.ax_3d.view_init(elev=10, azim=45)
        
        self.fig.canvas.draw()
    
    def create_controls(self):
        """Create control sliders and buttons"""
        # Control positions
        slider_height = 0.03
        slider_width = 0.15
        button_width = 0.08
        button_height = 0.04
        
        # Global adjustment sliders
        y_base = 0.15
        spacing = 0.05
        
        # Overall depth scale
        ax_depth_scale = plt.axes([0.1, y_base, slider_width, slider_height])
        self.slider_depth_scale = Slider(ax_depth_scale, 'Depth Scale', 0.1, 3.0, valinit=1.0)
        self.slider_depth_scale.on_changed(self.update_depth_scale)
        
        # Overall Y offset (up/down)
        ax_y_offset = plt.axes([0.3, y_base, slider_width, slider_height])
        self.slider_y_offset = Slider(ax_y_offset, 'Height Offset', -0.3, 0.3, valinit=0.0)
        self.slider_y_offset.on_changed(self.update_y_offset)
        
        # Body part specific adjustments
        y_base2 = y_base - spacing
        
        # Arm depth
        ax_arm_depth = plt.axes([0.1, y_base2, slider_width, slider_height])
        self.slider_arm_depth = Slider(ax_arm_depth, 'Arm Depth', -0.5, 0.5, valinit=0.0)
        self.slider_arm_depth.on_changed(self.update_arm_depth)
        
        # Leg depth
        ax_leg_depth = plt.axes([0.3, y_base2, slider_width, slider_height])
        self.slider_leg_depth = Slider(ax_leg_depth, 'Leg Depth', -0.3, 0.3, valinit=0.0)
        self.slider_leg_depth.on_changed(self.update_leg_depth)
        
        # Torso lean
        ax_torso_lean = plt.axes([0.5, y_base, slider_width, slider_height])
        self.slider_torso_lean = Slider(ax_torso_lean, 'Torso Lean', -0.3, 0.3, valinit=0.0)
        self.slider_torso_lean.on_changed(self.update_torso_lean)
        
        # Head position
        ax_head_depth = plt.axes([0.5, y_base2, slider_width, slider_height])
        self.slider_head_depth = Slider(ax_head_depth, 'Head Depth', -0.2, 0.2, valinit=0.0)
        self.slider_head_depth.on_changed(self.update_head_depth)
        
        # Control buttons
        button_y = 0.05
          # Reset button
        ax_reset = plt.axes([0.1, button_y, button_width, button_height])
        self.button_reset = Button(ax_reset, 'Reset')
        self.button_reset.on_clicked(self.reset_pose)
        
        # Save button
        ax_save = plt.axes([0.2, button_y, button_width, button_height])
        self.button_save = Button(ax_save, 'Save Pose')
        self.button_save.on_clicked(self.save_pose)
        
        # Apply constraints button
        ax_constraints = plt.axes([0.3, button_y, button_width * 1.5, button_height])
        self.button_constraints = Button(ax_constraints, 'Apply Constraints')
        self.button_constraints.on_clicked(self.apply_constraints)
        
        # View preset buttons
        ax_front = plt.axes([0.5, button_y, button_width, button_height])
        self.button_front = Button(ax_front, 'Front View')
        self.button_front.on_clicked(lambda x: self.set_view(0, 0))
        
        ax_side = plt.axes([0.6, button_y, button_width, button_height])
        self.button_side = Button(ax_side, 'Side View')
        self.button_side.on_clicked(lambda x: self.set_view(0, 90))
        
        ax_angle = plt.axes([0.7, button_y, button_width, button_height])
        self.button_angle = Button(ax_angle, '3/4 View')
        self.button_angle.on_clicked(lambda x: self.set_view(20, 45))
    
    def update_depth_scale(self, val):
        """Update overall depth scaling"""
        scale_factor = self.slider_depth_scale.val
        self.pose_3d_current[:, 2] = self.pose_3d_original[:, 2] * scale_factor
        self.plot_3d_pose()
    
    def update_y_offset(self, val):
        """Update overall Y (height) offset"""
        y_offset = self.slider_y_offset.val
        self.pose_3d_current[:, 1] = self.pose_3d_original[:, 1] + y_offset
        self.plot_3d_pose()
    
    def update_arm_depth(self, val):
        """Update arm depth positions"""
        arm_joints = [5, 6, 7, 8, 9, 10]  # shoulders, elbows, wrists
        depth_offset = self.slider_arm_depth.val
        
        for joint_idx in arm_joints:
            self.pose_3d_current[joint_idx, 2] = self.pose_3d_original[joint_idx, 2] + depth_offset
        
        self.plot_3d_pose()
    
    def update_leg_depth(self, val):
        """Update leg depth positions"""
        leg_joints = [11, 12, 13, 14, 15, 16]  # hips, knees, ankles
        depth_offset = self.slider_leg_depth.val
        
        for joint_idx in leg_joints:
            self.pose_3d_current[joint_idx, 2] = self.pose_3d_original[joint_idx, 2] + depth_offset
        
        self.plot_3d_pose()
    
    def update_torso_lean(self, val):
        """Update torso lean (forward/backward)"""
        torso_joints = [5, 6, 11, 12]  # shoulders and hips
        lean_offset = self.slider_torso_lean.val
        
        for joint_idx in torso_joints:
            self.pose_3d_current[joint_idx, 2] = self.pose_3d_original[joint_idx, 2] + lean_offset
        
        self.plot_3d_pose()
    
    def update_head_depth(self, val):
        """Update head position"""
        head_joints = [0, 1, 2, 3, 4]  # nose, eyes, ears
        depth_offset = self.slider_head_depth.val
        
        for joint_idx in head_joints:
            self.pose_3d_current[joint_idx, 2] = self.pose_3d_original[joint_idx, 2] + depth_offset
        
        self.plot_3d_pose()
    
    def reset_pose(self, event):
        """Reset pose to original"""
        self.pose_3d_current = self.pose_3d_original.copy()
        
        # Reset all sliders
        self.slider_depth_scale.reset()
        self.slider_y_offset.reset()
        self.slider_arm_depth.reset()
        self.slider_leg_depth.reset()
        self.slider_torso_lean.reset()
        self.slider_head_depth.reset()
        
        self.plot_3d_pose()
    
    def apply_constraints(self, event):
        """Apply anatomical constraints to current pose"""
        self.pose_3d_current = self.apply_anatomical_constraints(self.pose_3d_current)
        self.plot_3d_pose()
    
    def apply_anatomical_constraints(self, pose_3d):
        """Apply basic anatomical constraints"""
        pose_corrected = pose_3d.copy()
        
        # Ensure reasonable bone lengths
        # Hip to shoulder distance (torso)
        if np.any(pose_3d[11]) and np.any(pose_3d[5]):  # left hip to left shoulder
            torso_vector = pose_corrected[5] - pose_corrected[11]
            torso_length = np.linalg.norm(torso_vector)
            if torso_length > 0:
                # Normalize torso length
                ideal_torso = 0.6
                scale_factor = ideal_torso / torso_length
                pose_corrected[5] = pose_corrected[11] + torso_vector * scale_factor
        
        # Similar for right side
        if np.any(pose_3d[12]) and np.any(pose_3d[6]):
            torso_vector = pose_corrected[6] - pose_corrected[12]
            torso_length = np.linalg.norm(torso_vector)
            if torso_length > 0:
                ideal_torso = 0.6
                scale_factor = ideal_torso / torso_length
                pose_corrected[6] = pose_corrected[12] + torso_vector * scale_factor
        
        # Ensure feet are roughly at the same level (ground plane)
        if np.any(pose_3d[15]) and np.any(pose_3d[16]):  # both ankles
            avg_foot_y = (pose_corrected[15, 1] + pose_corrected[16, 1]) / 2
            pose_corrected[15, 1] = avg_foot_y
            pose_corrected[16, 1] = avg_foot_y
        
        return pose_corrected
    
    def set_view(self, elev, azim):
        """Set 3D view angle"""
        self.ax_3d.view_init(elev=elev, azim=azim)
        self.fig.canvas.draw()
    
    def save_pose(self, event):
        """Save current 3D pose"""
        pose_data = {
            'pose_2d': self.pose_2d.tolist(),
            'pose_3d_original': self.pose_3d_original.tolist(),
            'pose_3d_adjusted': self.pose_3d_current.tolist(),
            'joint_names': self.joint_names
        }
        
        filename = f"adjusted_pose_{len(os.listdir('.'))}.json"
        with open(filename, 'w') as f:
            json.dump(pose_data, f, indent=2)
        
        print(f"💾 Pose saved to: {filename}")
    
    def on_key_press(self, event):
        """Handle keyboard shortcuts"""
        if event.key == 'r':
            self.reset_pose(None)
        elif event.key == 's':
            self.save_pose(None)
        elif event.key == 'c':
            self.apply_constraints(None)
        elif event.key == '1':
            self.set_view(0, 0)    # Front
        elif event.key == '2':
            self.set_view(0, 90)   # Side
        elif event.key == '3':
            self.set_view(20, 45)  # 3/4 view


def load_pose_from_enhanced_results():
    """Load pose data from configured input image"""
    # Try to find MediaPipe and use it to get real 2D poses
    if MEDIAPIPE_AVAILABLE:
        # Use the configured input image
        image_path = Path(INPUT_IMAGE_PATH)
        
        if image_path.exists():
            print(f"📷 Loading pose from configured image: {image_path}")
        else:
            # Fallback: look for available images
            print(f"⚠️  Configured image '{INPUT_IMAGE_PATH}' not found. Looking for available images...")
            current_dir = Path(".")
            test_images = [f for f in current_dir.glob("*.jpg") if not f.name.endswith("_results.jpg")]
            test_images.extend([f for f in current_dir.glob("*.png") if not f.name.endswith("_results.png") and not f.name.endswith("_comparison.png")])
            
            if test_images:
                image_path = test_images[0]
                print(f"📷 Using fallback image: {image_path}")
            else:
                print("❌ No suitable images found!")
                return None, None, None
          # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return None, None, None
        
        # Initialize MediaPipe
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
          # Detect 2D pose
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # MediaPipe to COCO mapping
            coco_mapping = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
            
            pose_2d = np.zeros((17, 2))
            for i, mp_idx in enumerate(coco_mapping):
                if mp_idx < len(landmarks):
                    pose_2d[i, 0] = landmarks[mp_idx].x
                    pose_2d[i, 1] = landmarks[mp_idx].y
            
            # Generate initial 3D pose using simple method
            pose_3d = generate_initial_3d_pose(pose_2d)
            
            return pose_2d, pose_3d, image
    
    # Fallback: create sample pose
    print("📝 Creating sample pose for demonstration")
    pose_2d = np.array([
        [0.5, 0.15], [0.48, 0.12], [0.52, 0.12], [0.45, 0.10], [0.55, 0.10],
        [0.38, 0.25], [0.62, 0.25], [0.32, 0.40], [0.68, 0.40], [0.28, 0.55],
        [0.72, 0.55], [0.42, 0.60], [0.58, 0.60], [0.40, 0.80], [0.60, 0.80],
        [0.38, 0.95], [0.62, 0.95]
    ])
    
    pose_3d = generate_initial_3d_pose(pose_2d)
    
    return pose_2d, pose_3d, None


def generate_initial_3d_pose(pose_2d):
    """Generate initial 3D pose from 2D with reasonable depth estimates"""
    pose_3d = np.zeros((17, 3))
    
    # Copy X, Y coordinates (centered and flipped Y)
    pose_3d[:, 0] = pose_2d[:, 0] - 0.5  # Center X
    pose_3d[:, 1] = 0.5 - pose_2d[:, 1]  # Flip Y (up is positive)
    
    # Add depth estimates based on typical human anatomy
    depth_assignments = {
        'head': 0.0,      # Head at neutral depth
        'torso': -0.05,   # Torso slightly back
        'arms': 0.1,      # Arms forward
        'hands': 0.2,     # Hands further forward
        'hips': -0.03,    # Hips slightly back
        'legs': 0.05,     # Legs slightly forward
        'feet': 0.0       # Feet at neutral
    }
    
    # Apply depth by body part
    joint_groups = {
        'head': [0, 1, 2, 3, 4],
        'torso': [5, 6],
        'arms': [7, 8],
        'hands': [9, 10],
        'hips': [11, 12],
        'legs': [13, 14],
        'feet': [15, 16]
    }
    
    for group, depth in depth_assignments.items():
        if group in joint_groups:
            for joint_idx in joint_groups[group]:
                pose_3d[joint_idx, 2] = depth
    
    return pose_3d


def main():
    """Main function to run interactive pose adjuster"""
    print("🎮 Interactive 3D Pose Adjuster")
    print("=" * 40)
    print(f"📸 Input Image: {INPUT_IMAGE_PATH}")
    print("💡 To change input image, edit the INPUT_IMAGE_PATH variable at the top of this file")
    print()
    print("Controls:")
    print("  🖱️  Mouse: Rotate 3D view")
    print("  🎚️  Sliders: Adjust pose parameters")
    print("  ⌨️  Keyboard shortcuts:")
    print("     'r': Reset pose")
    print("     's': Save pose") 
    print("     'c': Apply constraints")
    print("     '1': Front view")
    print("     '2': Side view")
    print("     '3': 3/4 view")
    print()
    
    try:
        # Load pose data
        pose_2d, pose_3d, image = load_pose_from_enhanced_results()
        
        if pose_2d is None:
            print("❌ Failed to load pose data. Check your input image configuration.")
            return
        
        print("🚀 Starting interactive adjuster...")
        print("   Adjust the sliders to improve the 3D pose!")
        print("   The goal is to make the pose look more natural and anatomically correct.")
        
        # Create interactive adjuster
        adjuster = InteractivePoseAdjuster(pose_2d, pose_3d, image)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have some images in the directory and required packages installed.")


if __name__ == "__main__":
    main()
