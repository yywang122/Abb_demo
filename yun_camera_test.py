from cmath import sqrt
from this import d
import cv2
import numpy as np
import os
import pyrealsense2 as rs
import json
import sys
import serial
import time 
import runabb.abb as abb
import matplotlib.pyplot as plt




def get_aligned_images():
    frames = pipeline.wait_for_frames()  # 等待获取图像帧
    aligned_frames = align.process(frames)  # 获取对齐帧
    aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
    color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧

    ############### 相机参数的获取 #######################
    intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
    camera_parameters = {
    'fx': intr.fx, 'fy': intr.fy,
                         'ppx': intr.ppx, 'ppy': intr.ppy,
                         'height': intr.height, 'width': intr.width,
                         'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
                         }
    

    depth_image = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
    depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)  # 深度图（8位）
    depth_image_3d = np.dstack((depth_image_8bit, depth_image_8bit, depth_image_8bit))  # 3通道深度图
    color_image = np.asanyarray(color_frame.get_data())  # RGB图

    # 返回相机内参、深度参数、彩色图、深度图、齐帧中的depth帧
    return intr, depth_intrin, color_image, depth_image_3d, aligned_depth_frame

pipeline = rs.pipeline()  # 定义流程pipeline
config = rs.config()  # 定义配置config
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 配置depth流
config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)  # 配置color流
profile = pipeline.start(config)  # 流程开始
align_to = rs.stream.color  # 与color流对齐
align = rs.align(align_to)

def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image

def position_safe(RX,RY):
    "judge position for arm is safe "
    # if RX>750 or RX<400 or RY<-335 or RY>300:
    #     return False
    return True


def show_twoimg(img,img2):
    #im_h = cv2.hconcat([img,img2])
    #cv2.imshow(im_h)
    plt.figure()

    #subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(2,1) 

    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    axarr[0].imshow(img)
    cv2.waitKey()
    axarr[1].imshow(img2)
    cv2.waitKey()


object=list()
def display_instances(image_ori, boxes, masks, ids, names, scores):
    
    image_crop=image_ori[50:1000, 150:800]
    
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 950, 580)#(x,y)
    
    cv2.imshow('frame', image_crop)
    
    #cv2.waitKey(100)
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    
    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        if abs(x1-x2)<=50 and abs(y1-y2)<50:
            label = names[ids[i]]
            score = scores[i] if scores is not None else None
            caption = '{} {:.2f}'.format(label, score) if score else label
            mask = masks[:, :, i]

            image = apply_mask(image_ori, mask, color)
            # This adds the bounding box!! - NOTE
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            image = cv2.putText( image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
            #show_twoimg(image_ori,image)
            # cv2.imshow('frame', image_ori)
            image_crop2=image[50:1000, 150:800]
    
            cv2.namedWindow('MaskRCNN', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('MaskRCNN', 950, 580)#(x,y)
    
            cv2.imshow('MaskRCNN', image_crop2)
            #cv2.imshow('MaskRCNN', image)
            cv2.waitKey(100)
            #cv2.destroyAllWindows()

            #计算目标距离
            centerx = int((x1+x2)/2)
            centery = int((y1+y2)/2)
            intr, depth_intrin, rgb, depth, aligned_depth_frame = get_aligned_images()  # 获取对齐的图像与相机内参
            
            
            dis = aligned_depth_frame.get_distance(centerx, centery)  # （x, y)点的真实深度值
            print("distance:",dis)
            
            camera_coordinate = rs.rs2_deproject_pixel_to_point(intr, [centerx, centery], dis)
            # （x, y)点在相机坐标系下的真实值，为一个三维向量。
            # 其中camera_coordinate[2]仍为dis，camera_coordinate[0]和camera_coordinate[1]为相机坐标系下的xy真实距离。
            print(camera_coordinate)
            RX=float(1090*camera_coordinate[1]+477)
            RY=float(937.5*camera_coordinate[0]-38.125)
            
            global object
            # object.append((label,RX,RY,3))
            
            # print("object",object)
            
            # print("s1:",sorted(object, key = lambda s: s[1]))
            # print("s2",sorted(object, key = lambda s: s[2]))
            if object:
                for ob in object:
                    if abs(ob[1]-RX)<20 and abs(ob[2]-RY)<20:
                        ob[1]=ob[1]*ob[3]/(ob[3]+1)+RX/(ob[3]+1)
                        ob[2]=ob[2]*ob[3]/(ob[3]+1)+RY/(ob[3]+1)
                        ob[3]=ob[3]+1
                        break
                    if ob == object[-1]:
                        print("new ob")
                        object.append([label,RX,RY,3])
            else:
                print("new ob")
                object.append([label,RX,RY,3])
            object=sorted(object, key = lambda s: s[1])
            print("sorted objects:",object)
            # print(type(object),type(object[0][3]))
            # print(RX,RY)
            # print(position_safe(RX,RY))
            
            
                

            
            
    for ob in object:
        ob[3]=ob[3]-1

    for ob in object[:]:
        if ob[3]==0:
            object.remove(ob)
            print("del ob")
            
    print("")
    return image



if __name__ == '__main__':
    """
        test everything
    """
    import os
    import sys
    sys.path.insert(0, r'C:\Users\User\Desktop\gpu_Mask_RCNN-master\samples\coco')
    from mrcnn.config import Config
    from mrcnn import model as modellib,utils
     
    ROOT_DIR = os.getcwd()
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    #ser = serial.Serial('COM3',115200)
    R = abb.Robot(ip='192.168.125.1')
    R.set_cartesian([[264.88, -10.7, 708.8], [0,0,1,0]])
    #ser.write(serial.to_bytes([0x48, 0x49, 0x74, 0x01, 0x01, 0xA0, 0x01, 0x55, 0Xc6]))


    class BalloonConfig(Config):
        """Configuration for training on the toy  dataset.
        Derives from the base Config class and overrides some values.
        """
        # Give the configuration a recognizable name
        NAME = "balloon"

        # We use a GPU with 12GB memory, which can fit two images.
        # Adjust down if you use a smaller GPU.
        IMAGES_PER_GPU = 1

        # Number of classes (including background)
        NUM_CLASSES = 1 + 1  # Background + balloon

        # Number of training steps per epoch
        STEPS_PER_EPOCH = 100

        # Skip detections with < 90% confidence
        DETECTION_MIN_CONFIDENCE = 0.9

    config = BalloonConfig()
    

    model = modellib.MaskRCNN(mode="inference", model_dir='logs', config=config)

    # Load weights trained on MS-COCO
    # model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    model.load_weights('logs\cube20220808T1604\mask_rcnn_cube_0030.h5', by_name=True)
    class_names=['BG','cube']
    

    while True:
        intr, depth_intrin, frame, depth, aligned_depth_frame = get_aligned_images()  # 获取对齐的图像与相机内参

        
        result = model.detect([frame], verbose=0)
        
        frame = display_instances(frame, result[0]['rois'],  result[0]['masks'],  result[0]['class_ids'], class_names,result[0]['scores']
                     )
        
        
        '''
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #capture.release()
    cv2.destroyAllWindows()'''
    
    