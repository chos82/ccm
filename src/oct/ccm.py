'''
Created on 03.06.2026

@author: chris

C C M
======================
CamCaptureManipulation
======================
'''
import logging

import cv2 as cv
import numpy as np


SEPERATOR = '\n###################################################################\n'
__GREETING__  = SEPERATOR + '\nOpenCV CAM CAPTURE MANIPULATOR HAS STARTED\n' + SEPERATOR 
__STD_MSG__   = 'Oh shit. Something BAD happened :('

EYE = 'eye'
EYEGLASSES = 'eyeglasses'
FRONTALFACE = 'frontalcatface'
FRONTALFACE_EXTENDED = 'frontalcatface_extended'
FRONTALFACE_DEFAULT = 'frontalface_default'
FRONTALFACE_ALT = 'frontalface_alt'
FRONTALFACE_ALT_2 = 'frontalface_alt2'
FRONTALFACE_ALT_TREE = 'frontalface_alt_tree'
FULLBODY = 'fullbody'
LEFTEYE = 'lefteye'
LICENSE_PLATE_RUS = 'license_plate_rus'
LOWERBODY = 'lowerbody'
PROFILFACE = 'profileface'
RIGHTEYE = 'righteye'
RUSSIAN_PLATE_NUMBER = 'russian_plate_number'
SMILE = 'smile'
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
        if not isinstance(frame_part, np.ndarray):
            raise Exception('1st positional argument (frame_part) must be of type numpy.ndarray')
        self.frame_part = frame_part
        self.x = x   # Spalten-Offset (OpenCV x)
        self.y = y   # Zeilen-Offset  (OpenCV y)
        # FIX #6: shape[0]=Zeilen=Höhe, shape[1]=Spalten=Breite – konsistent mit OpenCV-Konvention
        self.h = frame_part.shape[0]
        self.w = frame_part.shape[1]
        
    def match_size(self, frame):
        '''
        Bettet self.frame_part in einen leeren Frame der Größe von `frame` ein.
        Die Position wird durch self.x (Spalte) und self.y (Zeile) bestimmt.
        Ist self.frame_part größer als frame, wird eine Exception geworfen.
        '''
        # Gleiche Größe → nichts zu tun
        if (frame.shape[0] == self.frame_part.shape[0] and
                frame.shape[1] == self.frame_part.shape[1]):
            return self.frame_part

        if (frame.shape[0] < self.frame_part.shape[0] or
                frame.shape[1] < self.frame_part.shape[1]):
            raise Exception(
                'cannot match size of a partial frame to size of a frame '
                'if partial frame is larger than frame'
            )

        # FIX #5: Leerer Ziel-Frame in voller Frame-Größe, dann frame_part einsetzen
        if frame.ndim == 2:
            result = np.zeros((frame.shape[0], frame.shape[1]), dtype=frame.dtype)
        else:
            result = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]), dtype=frame.dtype)

        row_start = self.y
        row_end   = self.y + self.frame_part.shape[0]
        col_start = self.x
        col_end   = self.x + self.frame_part.shape[1]

        result[row_start:row_end, col_start:col_end] = self.frame_part

        return result


'''
this class is mediating (mediator-pattern) from the OpenCV c-binaries to python-language
'''
class OCVDetector:
    classifier = cv.CascadeClassifier(cv.data.haarcascades + __haarcascades__[FRONTALFACE_DEFAULT])
    logger = None
    cap = None

    def __init__(self, logger=None, time_morphing=[]):
        self.logger = logger
        if logger is None:
            print('LOGGER MUST BE PROVIDED!!!')
        self.cap = cv.VideoCapture(0)
        
        cvFile = cv.__file__
        logger.info("successfully run 'import cv2 as cv', OpenCV's file is: " + cvFile)
        if self.cap is None:
            logger.error('video capture could not be initialized')
            exit(-1)
        if not self.cap.isOpened():
            logger.error("Cannot open camera")
            exit(-1)
        logger.info(__GREETING__)
        self.logger.info(FRONTALFACE_DEFAULT)
        if time_morphing is not None or time_morphing != []: 
            self.logger.info('time morphing is provided:')
            for e in time_morphing:
                self.logger.info('\t' + str(e))

    '''
    detects bounding boxes, according to selected ai-model
    '''
    def detect_bounding_box(self, vid):
        if not isinstance(vid, np.ndarray):
            raise Exception('OCVDetector.detect_bounding_box(): passed argument has wrong type')
        
        gray_image = cv.cvtColor(vid, cv.COLOR_BGR2GRAY)
        faces = self.classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        if len(faces) == 0:
            self.logger.info('no detectable found')
        else:
            self.logger.info(f'{len(faces)} detectable objects found')
            
        return faces

    def draw_bounding_box(self, frame, detected_objects, color=(0, 255, 0)):
        for (x, y, w, h) in detected_objects:
            cv.rectangle(frame, (x, y), (x + w, y + h), color, 4)

    def get_subImage(self, frame, obj, add_x=0, add_y=0):
        if not isinstance(frame, np.ndarray):
            raise Exception('OCVDetector.get_subImage(): passed argument has wrong type')
        if len(obj) != 4:
            raise Exception('OCVDetector.get_subImage(): passed argument must be exactly one rectangle object')
        
        x = int(int(obj[0]) - add_x * .5)
        y = int(int(obj[1]) - add_y * .5)
        w = int(int(obj[2]) + add_x)
        h = int(int(obj[3]) + add_y)

        # Koordinaten auf Frame-Grenzen klemmen
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)

        # FIX #3: OpenCV-Frame hat Form [Zeile, Spalte] → [y:y+h, x:x+w]
        frm_prt = frame[y:y+h, x:x+w]
        try:
            ret = PartialFrame(frm_prt, x, y)
        except Exception as e:
            raise Exception('could not construct PartialFrame') from e
        
        self.logger.info('extracted partial frame, x: {}, y: {}, w: {}, h: {}'.format(x, y, w, h))
        
        return ret

    def replace_subImage(self, frame, replacement):
        if not isinstance(frame, np.ndarray):
            raise Exception('OCVDetector.replace_subImage(): frame has wrong type')
        if not isinstance(replacement, PartialFrame):
            raise Exception('OCVDetector.replace_subImage(): replacement has wrong type')

        matched_part = replacement.match_size(frame)

        # FIX #6: Maske mit korrekter x/y-Semantik (x=Spalte, y=Zeile)
        mask = np.zeros(frame.shape[:2], dtype=bool)
        row_start = replacement.y
        row_end   = replacement.y + replacement.h
        col_start = replacement.x
        col_end   = replacement.x + replacement.w
        mask[row_start:row_end, col_start:col_end] = True

        if frame.ndim == 3:
            mask = mask[:, :, np.newaxis]  # Broadcasting auf alle Kanäle

        try:
            np.copyto(frame, matched_part, where=mask, casting='unsafe')
            self.logger.info('replaced a sub-frame')
        except Exception as e:
            raise Exception('could not copy replacement into frame (given a mask)') from e
        
        return frame

    '''
    executes the detection
    '''
    def run(self):
        i = 0
        cv.namedWindow('CCM', cv.WINDOW_KEEPRATIO)
        while True:
            i += 1
            
            try:
                frame = self.readCapture()
            except Exception:
                break

            # detect and draw bounding box
            detected_objects = self.detect_bounding_box(frame)
            self.draw_bounding_box(frame, detected_objects, (255, 0, 0))

            # FIX #4: Guard – kein Zugriff auf Index 0 wenn keine Objekte erkannt
            if len(detected_objects) == 0:
                cv.imshow('CCM', frame)
                if cv.waitKey(1) == ord('q'):
                    break
                continue

            # get black/white frame and thresholded frame from it
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            _ret, bw_frame = cv.threshold(gray_frame, 100, 255, cv.THRESH_BINARY)
            
            # find contours in thresholded frame and draw them
            contours, _hierarchy = cv.findContours(bw_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            frame = cv.drawContours(frame, contours, -1, (0, 255, 75), 2)

            obj = tuple(detected_objects[0])

            try:
                frame_part = self.get_subImage(bw_frame, obj, 100, 100)
            except Exception as e:
                raise Exception('could not obtain frame part') from e
            

            contours, _hierarchy = cv.findContours(bw_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            frame = cv.drawContours(frame, contours, -1, (0, 255, 75), 2)
            
            # try:
            #     frame = self.replace_subImage(frame, frame_part)
            # except Exception as e:
            #     self.logger.error(e)
            #     raise Exception('could not replace partial into frame') from e

            ####################################
            ## TODO: operations on sub_frame ###
            ####################################

            cv.imshow('CCM', frame)

            if cv.waitKey(1) == ord('q'):
                break

        self.logger.info(str(i) + ' times run')
        self.cap.release()
        cv.destroyAllWindows()
        return 0

    '''
    wrapper for OpenCV function
    '''
    def readCapture(self):
        ret, frame = self.cap.read()
        if not ret:
            raise Exception('OCVDetector.readCapture(): could not receive video stream')
        return frame
    
# --------class end-------
# end of class OCVDetector


def initLogger():
    logger = logging.getLogger('cam-cap-starter')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('cam-cap-starter.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def main():
    logger = initLogger()
    time_morph = lambda x, y, c: (x, y, c)
    Det = OCVDetector(logger, time_morphing=[time_morph])
    det = None
    i = 0
    logger.info('main(): event-loop')
    while True:
        try:
            det = Det.run()
            logger.info('executed OCVDetector.run()')
        except Exception:
            continue
        i += 1
        if det == 0:
            exit(-1)


if __name__ == '__main__':
    main()

