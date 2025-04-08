#http://www.pysource.com

from realsense_camera import RealsenseCamera
from object_detection import ObjectDetection
import cv2

# Create the Camera object
camera = RealsenseCamera()

# Create the Object Detection object
object_detection = ObjectDetection()


while True:
    # Get frame from realsense camera
    ret, color_image, depth_image = camera.get_frame_stream()
    height, width, _ = color_image.shape

    # Get the object detection
    bboxes, class_ids, score = object_detection.detect(color_image)
    for bbox, class_id, score in zip(bboxes, class_ids, score):
        x, y, x2, y2 = bbox
        color = object_detection.colors[class_id]
        cv2.rectangle(color_image, (x, y), (x2, y2), color, 2)

        # display name
        class_name = object_detection.classes[class_id]
        cv2.putText(color_image, f"{class_name}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Get center of the bbox
        cx, cy = (x + x2) // 2, (y + y2) // 2
        distance = camera.get_distance_point(depth_image, cx, cy)

        # Draw circle
        cv2.circle(color_image, (cx, cy), 5, color, -1)
        cv2.putText(color_image, f"Distance: {distance} cm", (cx, cy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # show color image
    cv2.imshow("Color Image", color_image)
    cv2.imshow("depth Image", depth_image)
    key = cv2.waitKey(1)
    if key == 27:
        break

# release the camera
camera.release()
cv2.destroyAllWindows()