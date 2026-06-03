'''
Created on 03.06.2026

@author: chris

C C M
======================
CamCaptureManipulation
======================
'''
import numpy as np
import cv2 as cv

import logging
import unittest

class OCVDetector:
    logger = logging.getLogger('cam-cap')
    face_classifier = cv.CascadeClassifier(
            cv.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def __init__(self):
        # create logger with 'spam_application'
        logger = logging.getLogger('cam-cap')
        logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler('cam-cap.log')
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        sep = '######################################'
        logger.info(sep)
        logger.info('starting cam capture manipulator')
        logger.info(sep)
        
        
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()


    def detect_bounding_box(self, vid):
        gray_image = cv.cvtColor(vid, cv.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        if(len(faces)==0):
            self.logger.info('no detectable found')
        else:
            n = str(len(faces))
            self.logger.info(n + ' detectable objects found')
            
        return faces

    def draw_bounding_box(self, frame, detected_objects):
        for (x,y,w,h) in detected_objects:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
        

# @TODO
# @returns largest bounding box of frames l1, l2
    def max_bb(self, l1, l2=[]):
        l = []
        if(isinstance(l1, np.ndarray)):
            l = l1
        if(isinstance(l2, np.ndarray)):
            l = l2
            
        x = l.shape[0]
        y = l.shape[1]
        
        if(isinstance(l1, np.ndarray) and isinstance(l2, np.ndarray)):
            self.logger.info('???????????')
            #TODO
        
        self.logger.info(x)
        
        return None

    def unwrap_numpy(self, a):
        if(not isinstance(a, np.ndarray)):
            raise Exception('type missmatch', 'f:unwrapping')
    
        self.logger.info("unwrap_numpy\n__________________\n__________________")
        self.logger.info(a[0])
        self.logger.info(a[0][0])
        t = 'variable to store type of function unwrap_numpy parameter'
        try:
            t=type(int(a[0][0]))
        except TypeError:
            m = a[0]
            n = a[0][0]
            self.logger.info('this is unwrap_numpy`s parameter a[0]:')
            self.logger.info(m)
            self.logger.info('this is unwrap_numpy`s parameter a[0][0]:')
            self.logger.info(n)
            s = a[0][0].shape
            self.logger.info('this is unwrap_numpy`s parameter a[0][0].scale:')
            self.logger.info(s)
            t = type(a[0][0].shape)
            self.logger.info('this is unwrap_numpy`s parameter a[0][0].scale.type:')
            self.logger.info(t)

            self.logger.error('coud not convert numpy ndarray to python scaLar')
            raise Exception('CRITICAL')
            
        
        
        #'CONTINUE'
        
        
        
        self.logger.info('ööööööö')
        self.logger.info(t)
    
        try:
            x = int(a[0][0])
            y = int(a[0][1])
            w = int(a[0][2])
            h = int(a[0][3])
        except:
            raise Exception('unwrap_numpyl104')
    
        self.logger.info('x,y,w,h')
        print(x, y, w, h)
        self.logger.info(x)
        self.logger.info('now its unwrapped')
        
        return (x,y,w,h)
    
    def run(self):
        while True:
            # Capture frame-by-frame
            ret, frame = self.cap.read()

            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # Our operations on the frame come here
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # detct and draw bounding box
            draw = []
            detected_objects = self.detect_bounding_box(frame)
    
            self.draw_bounding_box(frame, detected_objects)
    
            try:
                _f = self.unwrap_numpy(detected_objects)
            except:
                self.logger.error('\n\n\n   unwrapping of numpy array failed') #last line
                #raise Exception('unwrapping of numpy array failed')
                return -1
        
            draw += self.max_bb(frame)
            self.draw_bounding_box(frame, draw)
    
    
            # Find Canny edges 
            #edged = cv.Canny(frame, 30, 200) 
    
            # Finding Contours 
            # Use a copy of the image e.g. edged.copy()
            # since findContours alters the image 
            copy = gray.copy() 
            contours, hierarchy = cv.findContours(copy, 
                                  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) 
    
            # Draw all contours 
            # -1 signifies drawing all contours 
            cv.drawContours(copy, contours, -1, (0, 255, 0), 3) 
    
            # Display the resulting frame
            cv.imshow('Cam Capture Manipulation', frame)
            if cv.waitKey(1) == ord('q'):
                break
        
            # When everything done, release the capture
            self.cap.release()
            cv.destroyAllWindows()
        
        return
    
# --------class end-------
# end of class OCVDetector  
    
class OCVDetectorTest(unittest.TestCase):
    def testInitialization(self):
        Det = OCVDetector()
        self.assertTrue(Det is not None, 'successfully initialized an instance of OCVDetector')
        cap = Det.cap
        self.assertTrue(cap is not None, 'instance has a capture field')
        
    
        

