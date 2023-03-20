import cv2
import numpy
import utils

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

