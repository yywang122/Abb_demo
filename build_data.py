import cv2
import os

# Create a folder to save the images
save_folder = 'detect_image'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Create subfolders for train and val
train_folder = os.path.join(save_folder, 'train')
val_folder = os.path.join(save_folder, 'val')
if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(val_folder):
    os.makedirs(val_folder)
cap = cv2.VideoCapture(0)
# Initialize the frame counter
frame_counter = 0

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    # Display the frame
    cv2.imshow('Frame', frame)
    
    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF
    
    # If the 's' key is pressed, save the frame
    if key == ord('s'):
        # Save the frame to the save_folder
        filename = f'frame_{frame_counter:06d}.jpg'
        cv2.imwrite(os.path.join(save_folder, filename), frame)
        print(f'Saved frame {frame_counter} to {filename}')
        
        # Split the saved images into train and val folders
        if frame_counter % 2 == 0:
            os.replace(os.path.join(save_folder, filename), os.path.join(train_folder, filename))
            print(f'Moved {filename} to train folder')
        else:
            os.replace(os.path.join(save_folder, filename), os.path.join(val_folder, filename))
            print(f'Moved {filename} to val folder')
        
        # Increment the frame counter
        frame_counter += 1
    
    # If the 'q' key is pressed, exit the loop
    elif key == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()