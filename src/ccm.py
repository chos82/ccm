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
from test.test_signal import PidfdSignalTest

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

class PartialFrame:
    x = 0
    y = 0
    h = 0
    w = 0
    frame_part = None
    
    def __init__(self, frame_part, x, y):
        if(not isinstance(frame_part, np.ndarray)):
            raise Exception('1st positional argument (frame_part) must be of type numpy.ndarray')
        self.frame_part = frame_part
        self.x = x
        self.y = y
        self.w = frame_part.shape[0]
        self.h = frame_part.shape[1]
        
    def match_size(self, frame, x, y):
        '''
        tests if frame and self.frame_part do have the same size
        if so, do nothing. if not, self.frame_part will be enlarged to size of frame (x and y dimension)
        if self.frame_part is larger than frame an exception is raised
        '''
        # do frame and replacement have the same size?
        if(frame.shape[0] == self.frame_part.shape[0] and 
            frame.shape[1] == self.frame_part.shape[1]):
            return
        if(frame.shape[0] < self.frame_part.shape[0] or
            frame.shape[1] < self.frame_part.shape[1]):
            raise Exception('cannot match size of a partial frame to size of a frame if partial frame id larger than frame')
            
        empty_line = [None for _i in range(frame.shape[1])]


        def construct_line(start, stop):
            ret = []
            i = 0
            while i < frame.shape[1]:
                if(i < start or i >= stop):
                    ret.append(None)
                    i = i + 1
                else:
                    ret.extend(self.frame_part[i])
                    i = i + self.frame_part.shape[1]
            return ret
        
        result_frame = []
        
        for i in range(frame.shape[0]):
            if(i < x or i > x + self.frame_part.shape[0]):
                result_frame.append(empty_line)
            else:
                result_frame.append( construct_line(y, y + self.frame_part.shape[1]) )
        
        return np.array(result_frame)
        

'''
this class is mediating (mediator-pattern) from the OpenCV c-binaries to python-language
'''
class OCVDetector:
    classifier = cv.CascadeClassifier(cv.data.haarcascades + __haarcascades__[FRONTALFACE_DEFAULT])
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
        faces = self.classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        if(len(faces)==0):
            self.logger.info('no detectable found')
        else:
            n = str(len(faces))
            self.logger.info(n + ' detectable objects found')
            
        return faces

    def draw_bounding_box(self, frame, detected_objects, color=(0, 255, 0)):
        for (x,y,w,h) in detected_objects:
            cv.rectangle(frame, (x, y), (x + w, y + h), color, 4)
        
    def get_subImage(self, frame, obj, add_x=0, add_y=0):
        if(not isinstance(frame, np.ndarray)):
            raise Exception('OCVDetector.get_subImage(): passed argument has wrong type')
        if(not len(obj)==4):
            raise Exception('OCVDetector.get_subImage(): passed argument must be exactly one rectangle object')
        
        x = int(int(obj[0]) - add_x *.5)
        y = int(int(obj[1]) - add_y *.5)
        w = int(int(obj[2]) + add_x)
        h = int(int(obj[3]) + add_y)
        
        frm_prt = frame[x:x+w, y:y+h]
        ret = None
        try:
            ret = PartialFrame(frm_prt, x, y)
        except Exception as e:
            raise Exception('could not construct PartialFrame') from e
        
        return ret
    
    
    def replace_subImage(self, frame, replacement):
        if(not isinstance(frame, np.ndarray)):
            raise Exception('OCVDetector.get_subImage(): passed argument has wrong type')
        if(not isinstance(replacement, PartialFrame)):
            raise Exception('OCVDetector.get_subImage(): passed argument has wrong type')
        
        #rep = None
        
        def in_part_dim1(i, start, stop):
            if(i < start or i > stop):
                return False
            return True
        
        def in_part_dim0(i, start, stop):
            if(i < start or i > stop):
                return [False for _i in range(frame.shape[1])]
            return line
        
        line = [in_part_dim1(i, replacement.x, replacement.x + replacement.w) for i in range(frame.shape[1])]
        mask = [in_part_dim0(i, replacement.y, replacement.y + replacement.h) for i in range(frame.shape[0])]
        try:
            mask = np.array(mask)
        except Exception as e:
            raise Exception('could not construct numpy.array from mask') from e
        
        try:
            np.copyto(frame, replacement.frame_part, where=mask)
        except Exception as e:
            raise Exception('could not copy replacement into frame (given a mask)') from e
        
        return frame        
        
    '''
    executes the detection
    '''
    def run(self):
        i = 0
        while True:
            i = i+1
            
            try:
                frame = self.readCapture()
            except:
                break
            
            # Our operations on the frame come here

            # detect and draw bounding box
            detected_objects = self.detect_bounding_box(frame)
            self.draw_bounding_box(frame, detected_objects, (255, 0, 0))
            
            # get black/white frame and thresholded frame from it
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            _ret, bw_frame = cv.threshold(gray_frame, 100, 255, cv.THRESH_BINARY)
            
            # find contours in thresholded frame and draw them
            contours, _hierarchy  = cv.findContours(bw_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            
            frame_contour = cv.drawContours(frame, contours, -1, (0,255,75), 2)
            
            x = detected_objects[0][0]
            y = detected_objects[0][1]
            w = detected_objects[0][2]
            h = detected_objects[0][3]
            obj = (x,y,w,h)
           
            frame_part = None
            try:
                frame_contour = self.get_subImage(frame, obj, 50, 50)
            except Exception as e:
                raise Exception('could not obtain frame part') from e 
            
            frame_part = frame_contour
            
            try:
                frame = self.replace_subImage(frame, frame_part)
            except Exception as e:
                raise Exception('could not replace partial into frame') from e 
            
            ####################################
            ## TODO: operations on sub_frame ###
            ####################################
 
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
        
        
class PartialFrameTest(unittest.TestCase):
    frame = np.array(
            [[i for i in range(50)] for j in range(50)]
        )
    frame_part = np.array(
            [[i for i in range(20)] for j in range(30)]
        )
    pf = PartialFrame(frame_part, 10, 5)
    
    def testMatchSize(self):
        mf = self.pf.match_size(self.frame, self.pf.x, self.pf.y)
        self.assertTrue(mf is not None)
        self.assertEqual(mf.shape[0], self.frame.shape[0])
        self.assertEqual(mf.shape[1], self.frame.shape[1])
                
                
class ScriptTest(unittest.TestCase):
    def testMainProcedure(self):
        main()
        
    
        

