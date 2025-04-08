#import pyrealsense2 as rs
from object_distance_detection.object_detection_student import ObjectDetection
from object_distance_detection.realsense_camera import RealsenseCamera
import cv2

# Create the Camera object
camera = RealsenseCamera()

# Create the Object Detection object with YOLOv5 model
object_detection = ObjectDetection(weights_path='C:/Users/User/Desktop/yolov5/15.pt')

# Specify the target class name
target_class_name = 'box'

while True:
    # Get frame from RealSense camera
    ret, color_image, depth_image = camera.get_frame_stream()
    if not ret:
        break

    # Perform object detection
    bboxes, class_ids, scores = object_detection.detect(color_image, conf=0.25)

    for bbox, class_id, score in zip(bboxes, class_ids, scores):
        x1, y1, x2, y2 = bbox[:4].astype(int)  # Extract coordinates
        class_name = object_detection.classes[int(class_id)]

        if class_name.lower() == target_class_name.lower():
            color = object_detection.colors[int(class_id)]
            
            # Draw bounding box
            cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)

            # Display class name
            cv2.putText(color_image, f"{class_name}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            # Get center of the bounding box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Get the distance of the object
            distance = camera.get_distance_point(depth_image, cx, cy)

            # Draw center point and distance text
            cv2.circle(color_image, (cx, cy), 5, color, -1)
            cv2.putText(color_image, f"Distance: {distance} cm", (cx, cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # Show color image
    cv2.imshow("Color Image", color_image)
    # Optionally show depth image
    # cv2.imshow("Depth Image", depth_image)

    # Break loop on 'ESC' key
    if cv2.waitKey(1) == 27:  # ESC key to exit
        break

# Release the camera
camera.release()
cv2.destroyAllWindows()