import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IECore



class FacialLandmarkDetection:
    '''
    Class for the Facial Landmark Detection Model.
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
        self.input_name = next(iter(self.net.inputs))
        self.input_shape = self.net.inputs[self.input_name].shape
        self.output_name = next(iter(self.net.outputs))
        self.output_shape = self.net.outputs[self.output_name].shape
        




    def load_model(self):
        self.check_model()

        #log.info('Reading the Intermediate Representation')
        self.exec_net = self.core.load_network(network = self.net,device_name = self.device, num_requests=1)
        print("Network Loading")



    def predict(self, image):
        
        in_img = self.preprocess_input(image)
        outputs = self.exec_net.infer({self.input_name:in_img})
        coords = self.preprocess_output(outputs)
        h,w = image.shape[0],image.shape[1]
        coords = coords*np.array([w,h,w,h])
        coords = coords.astype(np.int32)
        #Left eye coordinates
        left_start_xmin = coords[0]-10
        left_start_ymin = coords[1]-10
        left_end_xmax = coords[0]+10
        left_end_ymax = coords[1]+10

        #Right Eye coordinates
        right_start_xmin = coords[2]-10
        right_start_ymin = coords[3]-10
        right_end_xmax = coords[2]+10
        right_end_ymax = coords[3]+10

        left_eye = image[left_start_ymin:left_end_ymax,left_start_xmin:left_end_xmax]
        right_eye = image[right_start_ymin:right_end_ymax,right_start_xmin:right_end_xmax]

        i_coords = [[left_start_xmin,left_start_ymin,left_end_xmax,left_end_ymax], [right_start_xmin,right_start_ymin,right_end_xmax,right_end_ymax]]

        return left_eye,right_eye,i_coords

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
        #process_img = image
        #n,c,h,w = self.input_shape
        #process_img = cv2.resize(process_img,(w,h),interpolation=cv2.INTER_AREA)
        #Chaning the channels
        #process_img = process_img.transpose((2,0,1))
        #process_img = process_img.reshape((n,c,h,w))

        #return process_img
        img_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        
        img_resized = cv2.resize(img_cvt, (self.input_shape[3], self.input_shape[2]))
        img_processed = np.transpose(np.expand_dims(img_resized,axis=0), (0,3,1,2))
        #print(img_processed.shape)
        return img_processed

    def preprocess_output(self,outputs):
        
        out = outputs[self.output_name][0]
        leye_x = out[0].tolist()[0][0]
        leye_y = out[1].tolist()[0][0]
        reye_x = out[2].tolist()[0][0]
        reye_y = out[3].tolist()[0][0]
        
        return (leye_x, leye_y, reye_x, reye_y)