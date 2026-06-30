'''
exceptions for project cam-capture manipulation with OpenCV
'''

class OCTException(Exception):
    def __init__(self, message = '', frameinfo = None):
        if frameinfo is not None:
            message = message + '\tin: {}\tline {}\n'.format(frameinfo.filename, frameinfo.lineno)
        super().__init__(message)
    

class OCVReadError(Exception):
    __message__ = 'OCVDetector.run(): could not get a opencv-capture'