'''
exceptions for project cam-capture manipulation with OpenCV
'''

from inspect import currentframe, getframeinfo


class OCTException(Exception):
    def __init__(self, message = ''):
        frameinfo = getframeinfo(currentframe())
        message = message + '\n\tin: {}\tline {}'.format(frameinfo.filename, frameinfo.lineno)
        super().__init__(message)
    

class OCVReadError(Exception):
    __message__ = 'OCVDetector.run(): could not get a opencv-capture'