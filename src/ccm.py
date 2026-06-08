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

SEPERATOR = '\n###################################################################\n'
__GREETING__  = SEPERATOR + '\nOpenCV CAM CAPTURE MANIPULATOR HAS STARTED\n' + SEPERATOR 
__STD_MSG__   = 'Oh shit. Something BAD happened :('

EYE = 'eye',
EYEGLASSES = 'eyeglasses',
FRONTALFACE = 'frontalcatface',
FRONTALFACE_EXTENDED = 'frontalcatface_extended',
FRONTALFACE_DEFAULT = 'frontalface_defaultl',
FRONTALFACE_ALT = 'frontalface_alt',
FRONTALFACE_ALT_2 = 'frontalface_alt2',
FRONTALFACE_ALT_TREE = 'frontalface_alt_tree',
FULLBODY = 'fullbody',
LEFTEYE = 'lefteye',
LICENSE_PLATE_RUS = 'license_plate_rus',
LOWERBODY = 'lowerbody',
PROFILFACE = 'profileface',
RIGHTEYE = 'righteye',
RUSSIAN_PLATE_NUMBER = 'russian_plate_number',
SMILE = 'smile',
UPPERBODY = 'upperbody'

__models__= [
    (EYE, 'haarcascade_eye.xml'),
    (EYEGLASSES, 'haarcascade_eye_tree_eyeglasses.xml'),
    (FRONTALFACE, 'haarcascade_frontalcatface.xml'),
    (FRONTALFACE_EXTENDED, 'haarcascade_frontalcatface_extended.xml'),
    (FRONTALFACE_DEFAULT, 'haarcascade_frontalface_default.xml'),
    (FRONTALFACE_ALT, 'haarcascade_frontalface_alt.xml'),
    (FRONTALFACE_ALT_2, 'haarcascade_frontalface_alt2.xml'),
    (FRONTALFACE_ALT_TREE, 'haarcascade_frontalface_alt_tree.xml'),
    (FULLBODY, 'haarcascade_fullbody.xml'),
    (LEFTEYE, 'haarcascade_lefteye_2splits.xml'),
    (LICENSE_PLATE_RUS, 'haarcascade_license_plate_rus_16stages.xml'),
    (LOWERBODY, 'haarcascade_lowerbody.xml'),
    (PROFILFACE, 'haarcascade_profileface.xml'),
    (RIGHTEYE, 'haarcascade_righteye_2splits.xml'),
    (RUSSIAN_PLATE_NUMBER, 'haarcascade_russian_plate_number.xml'),
    (SMILE, 'haarcascade_smile.xml'),
    (UPPERBODY, 'haarcascade_upperbody.xml')
]

__haarcascades__ = dict(__models__) 

'''
this class is mediating (mediator-pattern) from the OpenCV c-binaries to python-language
'''
class OCVDetector:
    classifier = cv.CascadeClassifier(cv.data.haarcascades + __haarcascades__[FRONTALFACE_DEFAULT])
    #classifier = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    logger = None
    cap = None

    def __init__(self, logger = None):
        self.logger = logger
        if(logger is None): print('LOGGER MUST BE PROVIDED!!!')
        self.cap = cv.VideoCapture(0)
        
        cvFile = cv.__file__
        logger.info("successfully run \'import cv2 as cv\', OpenCV`s file is: " + cvFile)
        if(self.cap == None):
            logger.error('video capture could not be initialized')
            exit(-1)
        if not self.cap.isOpened():
            logger.error("Cannot open camera")
            exit(-1)
        logger.info(__GREETING__)
        self.logger.info(FRONTALFACE_DEFAULT)
        
        
    '''
    detects bounding boxes, according to selected ai-modell
    '''
    def detect_bounding_box(self, vid):
        if(not isinstance(vid, np.ndarray)):
            raise Exception('OCVDetector.detect_bounding_box(): passed argument has wrong type')
        
        gray_image = cv.cvtColor(vid, cv.COLOR_BGR2GRAY)
        #faces = self.face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        faces = self.classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
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
        isNumpy_a = isinstance(l1, np.ndarray)
        isNumpy_b = isinstance(l2, np.ndarray)
        if(isNumpy_a):
            l = l1
        if(isNumpy_b):
            l = l2
        if(not(isNumpy_a and isNumpy_b)):
            raise Exception('ArgumentError')
            
        x = l[0].shape
        y = l.shape[1]
        
        if(isinstance(l1, np.ndarray) and isinstance(l2, np.ndarray)):
            self.logger.info('???????????')
            #TODO
        
        self.logger.info('shape of l[0]')
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
    
    '''
    executes the detection
    @return -1 indicates unsuccessful execution
    '''
    def run(self):
        i = 0
        while True:
            i = i+1
            
            # Capture frame-by-frame
            frame = self.readCapture()
            
            # Our operations on the frame come here
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # detect and draw bounding box
            draw = []
            detected_objects = self.detect_bounding_box(frame)
            self.draw_bounding_box(frame, detected_objects)
 
            cv.imshow('Cam Capture Manipulation', frame)
            if cv.waitKey(1) == ord('q'):
                break
        self.logger.info(str(i)+' times run')
        
        # When everything done, release the capture
        self.cap.release()
        cv.destroyAllWindows()
        
        return 0

    '''
    wrapper for OpenCV function
    '''
    def readCapture(self):
        # Capture frame-by-frame
        ret, frame = self.cap.read()
        # if frame is read correctly ret is True
        if not ret:
            raise Exception('OCVDetector.readCapture(): could not receive video stream')
        return frame

# --------class end-------
# end of class OCVDetector

def initLogger():
    # create logger with 'spam_application'
    logger = logging.getLogger('cam-cap-starter')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('cam-cap-starter.log')
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
    return logger

def main():
    logger = initLogger()
    Det = OCVDetector(logger)
    det = None
#    maxTries =  True
    i = 0
    logger.info('main(): event-loop')
    while(True):
        try:
            det = Det.run()
            logger.info('executed OCVDetector.run()')
        except:
            continue
        i = i+1
        if(det==0):
            exit(-1)
 
 
if __name__ == '__main__':
    main()
            
# ----start of tests-------
# the unit-tests begin here
    
class OCVDetectorTest(unittest.TestCase):
    Det = None
    def testInitialization(self):
        logger = initLogger()
        self.Det = OCVDetector(logger)
        self.assertTrue(self.Det is not None, 'successfully initialized an instance of OCVDetector')
        cap = self.Det.cap
        self.assertTrue(cap is not None, 'instance has a capture field')
        logger = self.Det.logger
        self.assertTrue(logger is not None, 'instance has been provided a logger')
        
    def testRun(self):
        if(self.Det is None): self.Det = self.testInitialization()
        self.Det.run()
        # TODO: no assertions yet!
        
    def testUnwrapPy(self):
        if(self.Det is None): self.Det = self.testInitialization()
        cap = self.Det.readCapture()
        boundingBox = self.Det.unwrap_numpy(cap)
        isTuple = isinstance(boundingBox, tuple)
        self.assertTrue(isTuple, 'OCVDetector.unwrapNumpy correctly returns a value of type tuple')
        print('shape:::::' + str(boundingBox.shape))
        
    def testDetectBoundingBox(self):
        if(self.Det is None): self.Det = self.testInitialization()
        detectedObject = self.Det.detect_bounding_box()
        t = type(detectedObject)
        print('type of detected objects') #remove
        print(t)                          #remove
        isTuple = isinstance(detectedObject, np.ndarray)
        self.assertTrue(isTuple, 'OCVDetector.detect_bounding_box correctly returns a value of type tuple')
        self.assertTrue(detectedObject.shape == 1)
                
        
        
class ScriptTest(unittest.TestCase):
    def testMainProcedure(self):
        main()
        
    
        

