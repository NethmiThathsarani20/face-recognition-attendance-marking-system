#!/usr/bin/env python3
"""
ESP32-CAM Live Demo Script

This script demonstrates the ESP32 cam working with live face detection.
Opens a window showing the ESP32 video feed with face detection boxes.
"""

import argparse
import sys
import os
import cv2

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.attendance_system import AttendanceSystem


def main():
    parser = argparse.ArgumentParser(
        description="ESP32-CAM Live Demo with Face Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show live stream with face detection
  python demo_esp32_live.py --url http://192.168.1.100:81/stream

  # Use local camera instead
  python demo_esp32_live.py --camera 0

  # Save video to file
  python demo_esp32_live.py --url http://192.168.1.100:81/stream --save output.avi
        """
    )
    
    parser.add_argument(
        '--url',
        help='ESP32 cam stream URL (e.g., http://192.168.1.100:81/stream)'
    )
    
    parser.add_argument(
        '--camera',
        type=int,
        help='Local camera index (e.g., 0, 1, 2)'
    )
    
    parser.add_argument(
        '--save',
        help='Save video to file (e.g., output.avi)'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Do not display video window (useful when saving only)'
    )
    
    args = parser.parse_args()
    
    # Determine camera source
    if args.url:
        camera_source = args.url
        print(f"📹 Using IP camera: {camera_source}")
    elif args.camera is not None:
        camera_source = args.camera
        print(f"📹 Using local camera: {camera_source}")
    else:
        print("❌ Error: Please specify --url or --camera")
        sys.exit(1)
    
    # Initialize attendance system
    print("🔄 Initializing face recognition system...")
    attendance_system = AttendanceSystem()
    print("✅ Face recognition system ready")
    
    # Open video capture
    print(f"📹 Opening video stream...")
    cap = cv2.VideoCapture(camera_source)
    
    if not cap.isOpened():
        print(f"❌ Failed to open camera: {camera_source}")
        sys.exit(1)
    
    print("✅ Video stream opened successfully")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    if fps == 0:
        fps = 20  # Default FPS if not available
    
    print(f"📊 Video: {width}x{height} @ {fps} FPS")
    
    # Setup video writer if saving
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(args.save, fourcc, fps, (width, height))
        print(f"💾 Saving video to: {args.save}")
    
    # Create window if displaying
    if not args.no_display:
        cv2.namedWindow('ESP32-CAM Live Demo', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('ESP32-CAM Live Demo', 960, 720)
    
    print("\n" + "="*80)
    print("🎥 LIVE STREAM STARTED")
    print("="*80)
    print("Controls:")
    print("  • Press 'q' or 'ESC' to quit")
    print("  • Press 's' to save current frame")
    print("  • Press SPACE to mark attendance for detected person")
    print("="*80 + "\n")
    
    frame_count = 0
    saved_frames = 0
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            
            if not ret or frame is None:
                print("❌ Failed to read frame")
                break
            
            frame_count += 1
            
            # Draw faces with detection and recognition
            processed_frame = attendance_system.draw_faces_with_names(frame)
            
            # Add frame counter overlay
            cv2.putText(
                processed_frame,
                f"Frame: {frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Display frame
            if not args.no_display:
                cv2.imshow('ESP32-CAM Live Demo', processed_frame)
            
            # Save frame if recording
            if writer:
                writer.write(processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("\n⏹️  Stopping stream...")
                break
            elif key == ord('s'):  # Save frame
                filename = f"esp32_frame_{frame_count}.jpg"
                cv2.imwrite(filename, processed_frame)
                saved_frames += 1
                print(f"💾 Saved frame to: {filename}")
            elif key == ord(' '):  # SPACE - Mark attendance
                print("\n📸 Marking attendance...")
                result = attendance_system.mark_attendance(frame, save_captured=True)
                if result['success']:
                    print(f"✅ Attendance marked: {result['user_name']} ({result['confidence']:.2f})")
                else:
                    print(f"❌ {result['message']}")
            
            # Show FPS every 30 frames
            if frame_count % 30 == 0:
                print(f"📊 Processed {frame_count} frames", end='\r')
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    
    finally:
        # Cleanup
        print("\n\n🧹 Cleaning up...")
        
        if cap:
            cap.release()
            print("✅ Camera released")
        
        if writer:
            writer.release()
            print(f"✅ Video saved to: {args.save}")
        
        if not args.no_display:
            cv2.destroyAllWindows()
            print("✅ Windows closed")
        
        print("\n" + "="*80)
        print("📊 SESSION SUMMARY")
        print("="*80)
        print(f"Total frames processed: {frame_count}")
        if saved_frames > 0:
            print(f"Frames saved manually: {saved_frames}")
        if args.save:
            print(f"Video recording: {args.save}")
        print("="*80)


if __name__ == '__main__':
    main()
