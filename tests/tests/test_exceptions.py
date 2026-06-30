'''
Created on 29.06.2026

@author: chris
'''
import unittest
import re

from oct.exceptions import OCTException


class Test(unittest.TestCase):


    def testName(self):
        exception = None
        try:
            self.thrower()
        except Exception as e:
            exception = e
        self.assertTrue(re.search(r"in.*\b*line\d*", str(exception)))

    def thrower(self):#
        raise OCTException

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()