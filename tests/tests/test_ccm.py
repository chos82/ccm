'''
Created on 29.06.2026

@author: chris
'''

import unittest
import numpy as np

from oct.ccm import OCVDetector, PartialFrame, initLogger


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
        if self.Det is None:
            self.testInitialization()
        self.Det.run()

    def testDetectBoundingBox(self):
        if self.Det is None:
            self.testInitialization()
        frame = self.Det.readCapture()
        detectedObject = self.Det.detect_bounding_box(frame)
        self.assertIsInstance(detectedObject, np.ndarray,
            'OCVDetector.detect_bounding_box correctly returns a value of type np.ndarray')


class PartialFrameTest(unittest.TestCase):
    frame = np.array(
        [[[i, i, i] for i in range(50)] for _j in range(50)], dtype=np.uint8
    )
    frame_part = np.array(
        [[[i, i, i] for i in range(20)] for _j in range(30)], dtype=np.uint8
    )
    pf = PartialFrame(frame_part, 10, 5)

    def testMatchSize(self):
        mf = self.pf.match_size(self.frame)
        self.assertIsNotNone(mf)
        self.assertEqual(mf.shape[0], self.frame.shape[0])
        self.assertEqual(mf.shape[1], self.frame.shape[1])

    def makeNumpyCube(self, rows, cols, channels):
        return np.zeros((rows, cols, channels), dtype=np.uint8)

    def testMatchSizeMultDim(self):
        cube = self.makeNumpyCube(15, 10, 3)
        cube_prt = self.makeNumpyCube(5, 5, 3)
        partial = PartialFrame(cube_prt, 3, 5)
        matched_partial = partial.match_size(cube)
        self.assertEqual(matched_partial.shape[0], cube.shape[0])
        self.assertEqual(matched_partial.shape[1], cube.shape[1])


class ScriptTest(unittest.TestCase):
    # automate pressing of 'q' for test
    # def testMainProcedure(self):
    #     main()
    pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()