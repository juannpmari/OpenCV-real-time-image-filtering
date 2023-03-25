import cv2
import numpy
import utils
import scipy.interpolate

#region channel mixing
def recolorRC(src,dst):
    """Simulate conversion from BGR to RC (red, cyan)
    The source and destination images must both be in BGR format
    Blues and greens are replaced with cyans.

    Pseudocode:
    dst.b = dst.g = 0.5*(src.b+src.g)
    dst.r = src.r
    """
    b, g, r = cv2.split(src)
    cv2.addWeighted(b,0.5,g,0.5,0,b)
    cv2.merge((b,b,r),dst)

def recolorRGV(src, dst):
    """ Simulate conversion from BGR to RGV (red, green, value)
        The source and destination images must both be in BGR format
        Blues are desaturated

        Pseudocode:
        dst.b = min(src.b,src.g,src.r)
        dst.g = src.g
        dst.r = src.r

    """
    b, g, r = cv2.split(src)
    cv2.min(b,g,b)
    cv2.min(b,r,b)
    cv2.merge((b,g,r),dst)

def recolorCMV(src, dst):
    """ Simulate conversion from BGR to CMV (cyan, magenta, value).
        The source and destination images must both be in BGR format.
        Yellows are desaturated.

        Pseudocode:
        dst.b = max(src.b, src.g, src.r)
        dst.g = src.g
        dst.r = src.r

    """
    b, g, r = cv2.split(src)
    cv2.max(b,g,b)
    cv2.max(b,r,b)
    cv2.merge((b,g,r),dst)

#endregion

#region curves
def createCurveFunc(points):
    """Return a function obtained interpolating control points. Each point is (channel_input, channel_output)"""
    if points is None:
        return None
    numPoints = len(points)
    if numPoints < 2:
        return None
    xs, ys = zip(*points)
    if numPoints < 4:
        kind = 'linear'
    # 'quadratic' is not implemented.
    else:
        kind = 'cubic'
    return scipy.interpolate.interp1d(xs, ys, kind, bounds_error = False)

class VFuncFilter(object):
    """A filter that applies a function to V channel, if gray-scale image, or all of BGR channels"""
    def __init__(self,vFunc=None,dtype=numpy.uint8):
        length = numpy.iinfo(dtype).max + 1
        self._vLookupArray = utils.createLookupArray(vFunc, length)
    
    def apply(self, src, dst):
        """Apply the filter with a BGR or gray source/destination"""
        srcFlatView = utils.flatView(src)
        dstFlatView = utils.flatView(dst)
        utils.applyLookupArray(self._vLookupArray, srcFlatView, dstFlatView)
    
class VCurveFilter(VFuncFilter):
    """A filter that applies a curve to V (or all of BGR)"""
    def __init__(self,vPoints,dtype = numpy.uint8):
        VFuncFilter.__init__(self,utils.createCurveFunc(vPoints),dtype)

class BGRFuncFilter(object):
    """A filter that applies different functions to each of BGR"""

    def __init__(self,vFunc=None,bFunc=None,gFunc=None,rFunc=None,dtype = numpy.uint8):
        """
            -vFunc: funct to be applied to all channels
            -bFunc: funct to be applied to b channel
            -gFunc: funct to be applied to g channel
            -rFunc: funct to be applied to r channel
        """
        length = numpy.iinfo(dtype).max + 1
        self._bLookupArray = utils.createLookupArray(utils.createCompositeFunc(bFunc,vFunc),length)
        self._gLookupArray = utils.createLookupArray(utils.createCompositeFunc(gFunc,vFunc),length)
        self._rLookupArray = utils.createLookupArray(utils.createCompositeFunc(rFunc,vFunc),length)
    
    def apply(self,src,dst):
        """Apply the filter with a BGR source/destination"""
        b, g, r = cv2.split(src)
        utils.applyLookupArray(self._bLookupArray, b, b)
        utils.applyLookupArray(self._gLookupArray, g, g)
        utils.applyLookupArray(self._rLookupArray, r, r)
        cv2.merge([b, g, r], dst)
    
class BGRCurveFilter(BGRFuncFilter):
    """A filter that applies different curves to each of BGR"""
    def __init__(self, vPoints = None, bPoints = None, gPoints = None, rPoints = None, dtype=numpy.uint8):
        """
            -vPoints: control points used to create curve funtion be applied to all channels
            -bPoints: control points used to create curve funtion be applied to b channel
            -gPoints: control points used to create curve funtion be applied to g channel
            -rPoints: control points used to create curve funtion be applied to r channel
        """
        BGRFuncFilter.__init__(self,utils.createCurveFunc(vPoints),utils.createCurveFunc(bPoints),
                               utils.createCurveFunc(gPoints),utils.createCurveFunc(rPoints),dtype)
        
class BGRPortraCurveFilter(BGRCurveFilter):
 """A filter that applies Portra-like curves to BGR."""
 
 def __init__(self, dtype = numpy.uint8):
    BGRCurveFilter.__init__(
    self,
    vPoints = [(0,0),(23,20),(157,173),(255,255)],
    bPoints = [(0,0),(41,46),(231,228),(255,255)],
    gPoints = [(0,0),(52,47),(189,196),(255,255)],
    rPoints = [(0,0),(69,69),(213,218),(255,255)],
    dtype = dtype)

#Other types of filters can be achieved with different control points

#endregion

#region edge detection
def strokeEdges(src, dst, blurKsize = 7, edgeKsize = 5):
    if blurKsize >= 3:
        blurredSrc = cv2.medianBlur(src, blurKsize)
        graySrc = cv2.cvtColor(blurredSrc,cv2.COLOR_BGR2GRAY)
    else:
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize = edgeKsize)
    normalizedInverseAlpha = (1.0/255)*(255-graySrc) #invert values to get black edges on white background
    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha #darken edges on the original BGR image
    cv2.merge(channels, dst)

#endregion

#region convolutional filters
class VConvolutionFilter(object):
    """A filter that applies a convolution to V (or all of BGR)"""
    def __init__(self,kernel):
        self._kernel = kernel
    def apply(self, src, dst):
        """Apply the filter with a BGR or gray source/destination"""
        cv2.filter2D(src, -1, self._kernel, dst)

class SharpenFilter(VConvolutionFilter):
    """A sharpen filter with a 1-pixel radius"""
    def __init__(self):
        kernel = numpy.array([[-1, -1, -1],[-1, 9, -1],[-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)

class FindEdgesFilter(VConvolutionFilter):
    """An edge-finding filter with a 1-pixel radius"""

    def __init__(self):
        kernel = numpy.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]])
        VConvolutionFilter.__init__(self,kernel)

class BlurFilter(VConvolutionFilter):
 """A blur filter with a 2-pixel radius."""
 
 def __init__(self):
    kernel = numpy.array([  [0.04, 0.04, 0.04, 0.04, 0.04],
                            [0.04, 0.04, 0.04, 0.04, 0.04],
                            [0.04, 0.04, 0.04, 0.04, 0.04],
                            [0.04, 0.04, 0.04, 0.04, 0.04],
                            [0.04, 0.04, 0.04, 0.04, 0.04]])
    VConvolutionFilter.__init__(self, kernel)

class EmbossFilter(VConvolutionFilter):
 """An emboss filter with a 1-pixel radius."""
 
 def __init__(self):
    kernel = numpy.array([  [-2, -1, 0],
                            [-1, 1, 1],
                            [ 0, 1, 2]])
    VConvolutionFilter.__init__(self, kernel)

#endregion