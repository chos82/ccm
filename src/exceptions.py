'''
exceptions for project cam-capture manipulation with OpenCV
'''

class OCVReadError(Exception):
    __message__ = 'OCVDetector.run(): could not get a opencv-capture'