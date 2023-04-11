import unittest

import numpy as np

from linear_regression.regression import (
        multi_regress
)

class TestMultiRegressValues(unittest.TestCase):
    
    def setUp(self):
        self.y = np.array([22.8,22.8,22.8,20.6,13.9,11.7,11.1,11.1])
        self.Z = np.array([[1,0],[1,2.3],[1,4.9],[1,9.1],[1,13.7],[1,18.3],[1,22.9],[1,27.2]])
    
    def testaoutput(self):
        a_exp = np.array([[23.715478],[-0.53784380]])
        print(a_exp)
        a_act,b,c = multi_regress(self.y, self.Z)
        print(a_act)
        self.assertTrue(np.allclose(a_exp, a_act))


if __name__ == "__main__":
    unittest.main()