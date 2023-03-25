import cv2
from managers import WindowManager, CaptureManager
import filters
import rects
from trackers import FaceTracker

class Cameo(object):

    def __init__(self):
        self._windowManager = WindowManager('Cameo',self.onKeypress)
        self._captureManager = CaptureManager(cv2.VideoCapture(0),self._windowManager,True)
        self._faceTracker = FaceTracker()
        self._shouldDrawDebugRects = False
        # self._curveFilter = filters.BGRPortraCurveFilter()
        # self._curveFilter = filters.EmbossFilter()
    
    def run(self):
        """Run the main loop"""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame

            self._faceTracker.update(frame)
            faces = self._faceTracker.faces
            rects.swapRects(frame, frame,[face.faceRect for face in faces])

            # filters.recolorCMV(frame,frame)

            # filters.strokeEdges(frame, frame)
            # self._curveFilter.apply(frame,frame)

            if self._shouldDrawDebugRects:
                self._faceTracker.drawDebugRects(frame)

            self._captureManager.exitFrame()
            self._windowManager.processEvents()
    
    def onKeypress(self, keycode):
        """Handle a keypress
        space -> take a screenshot
        tab -> Start/stop recording a screencast
        x -> start/stop drawing debug rectangles around faces
        escape -> quit
        """
        if keycode ==32: #space
            self._captureManager.writeImage('screenshot.png')
        elif keycode == 9: #tab
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo('screencast.avi')
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 120: #x
            self._shouldDrawDebugRects = not self._shouldDrawDebugRects
        elif keycode == 27: #escape
            self._windowManager.destroyWindow()

if __name__=="__main__":
    Cameo().run()





#Links a leer
# https://realpython.com/python-property/#the-getter-and-setter-approach-in-python
# https://www.programiz.com/python-programming/property
# https://en.wikipedia.org/wiki/Callback_(computer_programming)#:~:text=In%20computer%20programming%2C%20a%20callback,as%20part%20of%20its%20job.
# https://medium.com/understand-the-python/understanding-the-asterisk-of-python-8b9daaa4a558

