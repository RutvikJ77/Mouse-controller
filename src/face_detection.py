import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IECore



class FaceDetection:
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
        
        self.input_name = next(iter(self.net.inputs))
        self.input_shape = self.net.inputs[self.input_name].shape
        self.output_name = next(iter(self.net.outputs))
        self.output_shape = self.net.outputs[self.output_name].shape
        




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



    def predict(self, image,threshold):
        
        in_img = self.preprocess_input(image)
        outputs = self.exec_net.infer({self.input_name:in_img})
        coords = self.preprocess_output(outputs,threshold)
        coords = coords[0] #Edge Case
        h=image.shape[0]
        w=image.shape[1]
        coords = coords* np.array([w, h, w, h])
        coords = coords.astype(np.int32)
        cropped_face = image[coords[1]:coords[3],coords[0]:coords[2]]       
        return coords,cropped_face 

    def check_model(self):
        # Check for the supporting Layers
        supp_layers = self.core.query_network(network = self.net,device_name = self.device)
        unsupp_layers  = [l for l in self.net.layers.keys() if l not in supp_layers]
        if (len(unsupp_layers)!=0 and (self.extensions and 'CPU' in self.device)):
            self.core.add_extension(self.extensions,self.device)
            #log.info('Extension added to resolve the unsupported layers')
            if len(unsupp_layers)>0:
                print("ERROR: Unsupported Layers")
                exit(-1)

    def preprocess_input(self, image):
        process_img = image
        n,c,h,w = self.input_shape
        
        process_img = cv2.resize(process_img,(w, h))
        #Chaning the channels
        process_img = process_img.transpose((2,0,1))
        process_img = process_img.reshape((n,c,h,w))
        
        return process_img
        #image_resized = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        #img_processed = np.transpose(np.expand_dims(image_resized,axis=0), (0,3,1,2))
        #return img_processed       

    def preprocess_output(self,outputs,threshold):
        #Used for drawing boxes around face
        det = []
        out = outputs[self.output_name][0][0]
        for o in out:
            if o[2]>threshold:
                det.append([o[3],o[4],o[5],o[6]])

        return det
