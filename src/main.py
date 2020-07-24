
from face_detection import FaceDetection
from head_pose_estimation import Head_Pose_estimation
from facial_landmarks_detection import FacialLandmarkDetection
from gaze_estimation import GazeEstimation
from argparse import ArgumentParser
from input_feeder import InputFeeder
from mouse_controller import MouseController

import cv2
import os
import logging 
import numpy as numpy

def build_parser():
    #Parse command line arguments.

    #:return: command line arguments
    parser = ArgumentParser()
    parser.add_argument("-f", "--facedetectionmodel", required=True, type=str,
                        help="Specify Path to .xml file of Face Detection model.")
    parser.add_argument("-fl", "--faciallandmarkmodel", required=True, type=str,
                        help="Specify Path to .xml file of Facial Landmark Detection model.")
    parser.add_argument("-hp", "--headposemodel", required=True, type=str,
                        help="Specify Path to .xml file of Head Pose Estimation model.")
    parser.add_argument("-g", "--gazeestimationmodel", required=True, type=str,
                        help="Specify Path to .xml file of Gaze Estimation model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Specify Path to video file or enter cam for webcam")
    parser.add_argument("-flags", "--previewFlags", required=False, nargs='+',
                        default=[],
                        help="Specify the flags from fd, fld, hp, ge like --flags fd hp fld (Seperate each flag by space)"
                             "for see the visualization of different model outputs of each frame," 
                             "fd for Face Detection, fld for Facial Landmark Detection"
                             "hp for Head Pose Estimation, ge for Gaze Estimation." )
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-prob", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="Probability threshold for model to detect the face accurately from the video frame.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    
    return parser

def main():
    #Building the arguments
    args = build_parser().parse_args()
    previewFlag = args.previewFlags

    log = logging.getLogger()
    input_path = args.input
    inputFeed = None

    if input_path.lower()=='cam':
        inputFeed = InputFeeder('cam')
    else:
        if not os.path.isfile(input_path):
            log.error("Unable to find the input file specified.")
            exit(1)
        inputFeed = InputFeeder('video',input_path)
    
    #Creating Model paths
    model_path = {'FaceDetectionModel':args.facedetectionmodel, 'FacialLandmarksDetectionModel':args.faciallandmarkmodel, 
    'GazeEstimationModel':args.gazeestimationmodel, 'HeadPoseEstimationModel':args.headposemodel}

    for fnameKey in model_path.keys():
        if not os.path.isfile(model_path[fnameKey]):
            log.error('Unable to find the specified '+ fnameKey + 'binary file(.xml)')
            exit(1)

    #Creating Model Instances
    fd = FaceDetection(model_path['FaceDetectionModel'],args.device,args.cpu_extension)
    flm = FacialLandmarkDetection(model_path['FacialLandmarksDetectionModel'],args.device,args.cpu_extension) 
    gm = GazeEstimation(model_path['GazeEstimationModel'], args.device, args.cpu_extension)
    hpe = Head_Pose_estimation(model_path['HeadPoseEstimationModel'], args.device, args.cpu_extension)

    m_control = MouseController('medium','fast')

    #Loading data
    inputFeed.load_data()
    fd.load_model()
    flm.load_model()
    hpe.load_model()
    gm.load_model()


    
    frame_count = 0
    for ret,frame in inputFeed.next_batch():
        if not ret:
            break
        frame_count +=1
        if frame_count%10==0:
            cv2.imshow('Original Video',cv2.resize(frame,(500,500)))

        key = cv2.waitKey(60)
        coords,img = fd.predict(frame,args.prob_threshold)
        if type(img) == int:
            log.error("No face detected")
            if key ==27:
                break
            continue

        hpout = hpe.predict(img)
        left_eye,right_eye,eye_coord = flm.predict(img)
        mouse_coord, gaze_vec = gm.predict(left_eye,right_eye,hpout)

        if (not len(previewFlag)==0):
            preview_img = img
            if 'fd' in previewFlag:
                preview_img = img
            if 'fld' in previewFlag:
                start_l = (eye_coord[0][0]-10, eye_coord[0][1]-10)
                end_l = (eye_coord[0][2]+10,eye_coord[0][3]+10)
                start_r = (eye_coord[1][0]-10, eye_coord[1][1]-10)
                end_r = (eye_coord[1][2]+10,eye_coord[1][3]+10)
                cv2.rectangle(img,start_l,end_l,(0,255,0),2)
                cv2.rectangle(img,start_r,end_r,(0,255,0),2)
            if 'hp' in previewFlag:
                cv2.putText(preview_img,"Pose Angles: yaw:{:.2f} | pitch:{:.2f} | roll:{:.2f}".format(hpout[0],hpout[1],hpout[2]),(10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.25, (255, 255, 255), 1)
            if 'ge' in previewFlag:
                x,y,w = int(gaze_vec[0]*12),int(gaze_vec[1]*12),160
                lefteye = cv2.line(left_eye,(x-w,y-w),(x+w, y+w), (100,0,255), 1)
                cv2.line(lefteye,(x-w, y+w), (x+w, y-w), (100,0,255), 1)
                righteye = cv2.line(right_eye,(x-w, y-w), (x+w, y+w), (100,0,255), 1)
                cv2.line(righteye,(x-w, y+w), (x+w, y-w), (100,0,255), 1)
                img[eye_coord[0][1]:eye_coord[0][3],eye_coord[0][0]:eye_coord[0][2]] = lefteye
                img[eye_coord[1][1]:eye_coord[1][3],eye_coord[1][0]:eye_coord[1][2]] = righteye
            
            cv2.imshow("Detections",cv2.resize(preview_img,(500,500)))
        if frame_count%10==0:
            m_control.move(mouse_coord[0],mouse_coord[1])
        if key == 27:
            break
    log.error("Videostream Completed")
    cv2.destroyAllWindows()
    inputFeed.close()


if __name__ == '__main__':
    main()