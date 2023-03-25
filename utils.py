import cv2
import numpy
import scipy.interpolate

#region curves
def createCurveFunc(points):
    """Return a function derived from control points"""
    if points is None:
        return None
    numPoints = len(points)
    if numPoints < 2:
        return None
    xs, ys = zip(*points)
    if numPoints < 4:
        kind = 'linear'
        #quadratic is not implemented
    else:
        kind = 'cubic'
    return scipy.interpolate.interp1d(xs, ys, kind)#, bound_error = False)

def createLookupArray(func, length = 256):
    """Return a lookup (LUT) for whole-number inputs to a function
        The lookup values are clamped to [0, length -1]
        func: function returned by createCurveFunc
    """
    if func is None:
        return None
    lookupArray = numpy.empty(length)
    i=0
    while i < length:
        func_i = func(i)
        lookupArray[i] = min(max(0, func_i), length - 1) #Saturate values to [0,255]
        i += 1
    return lookupArray

def applyLookupArray(lookupArray,src,dst):
    """Map a source to a destination using a lookup."""
    if lookupArray is None:
        return
    dst[:] = lookupArray[src]

def createCompositeFunc(func0, func1):
    """Return a composite of two functions. Used to apply two curves simultaneously"""
    if func0 is None:
        return func1
    if func1 is None:
        return func0
    return lambda x: func0(func1(x))

def createFlatView(array):
    """Return a 1D view of an array of any dimensionality"""
    flatView = array.view()
    flatView.shape = array.size #array view equals number of element, to create a 1D view
    return flatView

#endregion


def isGray(image):
    """Return True if the image has one channel per pixel"""
    return image.ndim < 3

def widthHeightDividedBy(image,divisor):
    """Return an image's dimensions, divided by a value"""
    h, w = image.shape[:2]
    return (w/divisor, h/divisor)