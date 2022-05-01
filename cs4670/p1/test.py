import unittest
import numpy as np
import p1 as student
from scipy.ndimage import gaussian_filter, convolve, rotate

class ImreadTestCase(unittest.TestCase):
    def setUp(self):
        self.file = 'test.png'

    def testOutput(self):
        output = student.imread(self.file)
        self.assertEqual(output.shape, (500, 500, 3), 'Incorrect output size')
        t = output.dtype
        self.assertEqual(output.dtype.kind, 'f', 'Not a floating point array')
        min_val = np.min(output)
        max_val = np.max(output)
        self.assertTrue(0<=min_val and max_val<=1, 'Values not between 0 and 1')


class ConvolutionTestCase(unittest.TestCase):
    def setUp(self):
        self.random_img_2d = np.random.randn(101,101)
        self.random_img_3d = np.random.randn(100,100,3)
        self.random_filt = np.random.randn(3,5)

    def testSize(self):
        output = student.convolve(self.random_img_2d, self.random_filt)
        self.assertEqual(output.shape, (101,101), 'Incorrect output size')
        output = student.convolve(self.random_img_3d, self.random_filt)
        self.assertEqual(output.shape, (100,100,3), 'Incorrect output size')

    def testValues(self):
        output = student.convolve(self.random_img_2d, self.random_filt)
        output_gt = convolve(self.random_img_2d, self.random_filt, mode='constant')
        self.assertTrue(np.allclose(output, output_gt), 'Incorrect values')
        output = student.convolve(self.random_img_3d, self.random_filt)
        output_gt = convolve(self.random_img_3d, self.random_filt[:,:,None], mode='constant')
        self.assertTrue(np.allclose(output, output_gt), 'Incorrect values')

    def testValuesFlipped(self):
        output = student.convolve(self.random_filt, self.random_img_2d)
        output_gt = convolve(self.random_filt, self.random_img_2d, mode='constant')
        self.assertTrue(np.allclose(output, output_gt), 'Incorrect values when performing convolution with a filter larger than the image')


class GaussianTestCase(unittest.TestCase):
    def setUp(self):
        self.random_img = np.random.randn(100,100)

    def testSum(self):
        output = student.gaussian_filter(5,1)
        self.assertTrue(np.allclose(output.sum(),1),'Filters must sum to 1.')

    def testValues(self):
        filt = student.gaussian_filter(5,1)
        out = convolve(self.random_img, filt, mode='constant')
        out_gt = gaussian_filter(self.random_img, sigma=1, truncate=2, mode='constant')
        self.assertTrue(np.allclose(out, out_gt))


class GradientTestCase(unittest.TestCase):
    def setUp(self):
        self.img = student.imread('test.png')

    def testValues(self):
        mag, ori = student.gradient(self.img)
        dmag = np.diagonal(mag)[[0,69,275,420,425]]
        dori = np.diagonal(ori)[[0,69,275,420,425]]
        test_mag = np.array([0.46888004, 0.22153629, 0.14217979, 0., 0.0605548])
        test_ori = np.array([0.78539816,  3.14159265, -2.35619449,  0., 0.78539816])
        self.assertTrue(np.allclose(dmag, test_mag, rtol=1e-5),'Gradient magnitude does not match')
        self.assertTrue(np.allclose(dori, test_ori, rtol=1e-5),'Gradient orientation does not match')


class Check_distance_from_lineTestCase(unittest.TestCase):
    def setUp(self):
        self.theta = -1.57079633
        self.c = 274.89321881 
        self.x = np.array([10,100,300,0,100,100,100])
        self.y = np.array([275,275,275,0,100,274,276])
    
    def testValues(self):
        out = student.check_distance_from_line(self.x, self.y, self.theta, self.c, thresh=0.3)
        self.assertTrue(np.all(out[:3]==True) and np.all(out[3:]==False),'Check distance from line incorrect.')


class Draw_linesTestCase(unittest.TestCase):
    def setUp(self):
        self.img = student.imread('test.png')
        self.lines = [[ -1.57079633, 274.89321881],[ -1.02101761, -11.10678119],[  3.14159265,  68.89321881]]
    
    def testValues(self):
        out = student.draw_lines(self.img, self.lines, thresh=0.5)
        red_idx = list(zip(*np.where(np.all(out == np.array([1,0,0]), axis=-1))))
        self.assertTrue((275,100) in red_idx and (275,300) in red_idx,'Missing red line.')
        self.assertTrue((51, 104) in red_idx and (156, 276) in red_idx,'Missing red line.')
        self.assertTrue((100, 69) in red_idx and (300, 69) in red_idx,'Missing red line.')

def matches(list1, list2, rtol=1e-03):
    m = 0
    for l in list1:
        for sl in list2:
            if  np.allclose(l,sl,rtol=rtol):
                m += 1
    return m


class Hough_votingandLocalmaxTestCase(unittest.TestCase):
    def setUp(self):
        self.img = student.imread('test.png')
        self.gradmag, self.gradori = student.gradient(self.img)
        self.thetas = np.arange(-np.pi-np.pi/40, np.pi+np.pi/40, np.pi/40)
        imgdiagonal = np.sqrt(self.img.shape[0]**2 + self.img.shape[1]**2)
        self.cs = np.arange(-imgdiagonal, imgdiagonal, 2)
        self.votes = student.hough_voting(self.gradmag, self.gradori, self.thetas, self.cs, 0.1, 0.3, np.pi/40)

    def testHough(self):
        idx = list(zip(*np.where(self.votes>100)))
        sol_idx = [(21, 491),(21, 603),(41, 141),(41, 353),(61, 141),(61, 353),(81, 388),(81, 389),(81, 491),(81, 603)]
        m = matches(idx, sol_idx, 0)
        self.assertTrue(m>=7, 'At least seven large cell values (>100 votes) were not found.')
        self.assertTrue(len(idx)<20, 'Too many large cell values (>100 votes).')

    def testLocalmax(self):
        lines = student.localmax(self.votes, self.thetas, self.cs, 20, 15)
        sol_lines = [(-1.5707963267948966, 274.89321881345245),(-1.5707963267948966, 498.89321881345245),(-1.0210176124166828, -11.10678118654755),(0.0, -425.10678118654755),(0.0, -1.1067811865475505),(1.0210176124166828, -215.10678118654755),(1.570796326794897, -425.10678118654755),(1.570796326794897, -1.1067811865475505),(3.1415926535897936, 68.89321881345245),(3.1415926535897936, 274.89321881345245),(3.1415926535897936, 498.89321881345245)]
        m = matches(lines, sol_lines)
        self.assertTrue(m>=7, 'At least seven lines were not found.')
        self.assertTrue(len(lines)<20, 'Too many lines found.')


if __name__ == '__main__':
    unittest.main()
