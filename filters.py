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
    cv2.min(b,g,b)
    cv2.min(b,r,b)
    cv2.merge((b,g,r),dst)

#endregion

#region curves
def createCurveFunc(points):
    """Return a function derived from control points."""
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
    """A filter that applies a function to V (or all of BGR)"""
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
    """A filter that applies different functions to each og BGR"""

    def __init__(self,vFunc=None,bFunc=None,gFunc=None,rFunc=None,dtype = numpy.uint8):
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

#endregion

#region edge detection
#Seguir acá (último 20/3 - pag. 51 )
#endregion