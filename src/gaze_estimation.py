import cv2
import numpy as np
import logging as log
import math
from openvino.inference_engine import IECore



class GazeEstimation:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        
        self.exec_net = None
        self.device = device
        self.extensions = extensions
        self.infer_request = None
        self.core = None
        if not self.core:
            #log.info('Intializing plugin for {} device'.format(self.device))
            self.core =IECore()
        else:
            self.core=None

        self.model_weights = model_name.split('.')[0]+'.bin'
        self.model_structure = model_name
        self.net = self.core.read_network(model = self.model_structure,weights = self.model_weights)

        self.input_name = [i for i in self.net.inputs.keys()]
        self.input_shape = self.net.inputs[self.input_name[1]].shape
        self.output_name = [i for i in self.net.outputs.keys()]
        
        




    def load_model(self):
        #if not self.core:
            #log.info('Intializing plugin for {} device'.format(self.device))
            #self.core =IECore()
        #else:
            #self.core=None
        self.check_model()

        #log.info('Reading the Intermediate Representation')
        self.exec_net = self.core.load_network(network = self.net,device_name = self.device, num_requests=1)
        print("Network Loading")



    def predict(self,left_image,right_image, hpa):
        
        left_process =  self.preprocess_input(left_image)
        right_process = self.preprocess_input(right_image)
        outputs = self.exec_net.infer({'head_pose_angles':hpa, 'left_eye_image':left_process, 'right_eye_image':right_process})
        new_coord ,gaze_vec = self.preprocess_output(outputs,hpa)

        return new_coord,gaze_vec

    def check_model(self):
        # Check for the supported Layers
        supp_layers = self.core.query_network(network = self.net,device_name = self.device)
        unsupp_layers  = [l for l in self.net.layers.keys() if l not in supp_layers]
        if (len(unsupp_layers)!=0 and (self.extensions and 'CPU' in self.device)):
            self.core.add_extension(self.extensions,self.device)
            #log.info('Extension added to resolve the unsupported layers')
            if len(unsupp_layers)>0:
                print("ERROR: Unsupported Layers")
                exit(-1)

    def preprocess_input(self, image):
        in_img = image
        #print(image.shape)
        n,c,h,w = self.input_shape
        #print(n,c,h,w)
        in_img = cv2.resize(image,(w,h),interpolation=cv2.INTER_AREA)
        in_img= in_img.transpose((2,0,1))
        in_img=in_img.reshape((n,c,h,w))
        #right_resize = cv2.resize(right_image,(self.input_shape[3],self.input_shape[2]))
        
        
        #right_process = np.transpose(np.expand_dims(right_resize,axis=0),(0,3,1,2))
        return in_img

    def preprocess_output(self,outputs,hpa):
        gaze_vector = outputs[self.output_name[0]].tolist()[0]
        #gaze_vector = outputs[0]
        #rollval = gaze_vector[2]
        #gaze_vector = gaze_vector / np.linalg.norm(gaze_vector)
        rollval = hpa[2]
        cosval = math.cos(rollval*math.pi/180.0)
        sinval = math.sin(rollval * math.pi/180.0)

        new_x = gaze_vector[0] * cosval +gaze_vector[1]*sinval
        new_y = -gaze_vector[0] * sinval+gaze_vector[1]*cosval
        return (new_x,new_y) , (gaze_vector)
