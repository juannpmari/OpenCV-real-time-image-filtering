import cv2
from managers import WindowManager, CaptureManager
import filters
from filters import ChannelMixing
import rects
from trackers import FaceTracker

class Cameo(object):

    def __init__(self):
        self._windowManager = WindowManager('Cameo',self.onKeypress)
        self._captureManager = CaptureManager(cv2.VideoCapture(0),self._windowManager,False)
        self._channel_mixing_filter = ChannelMixing()
        self._faceTracker = FaceTracker()
        self._shouldDrawDebugRects = False
        self._curveFilter = [filters.BGRPortraCurveFilter(),
                             filters.EmbossFilter(),filters.SharpenFilter(),filters.FindEdgesFilter(),filters.BlurFilter() ]
        self._curveFilterNum = 0
        self._enable_edge_detection = False
    
    def run(self):
        """Run the main loop"""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame

            #Face tracking
            self._faceTracker.update(frame)
            faces = self._faceTracker.faces
            rects.swapRects(frame, frame,[face.faceRect for face in faces])

            #Filtering
            self._channel_mixing_filter.apply(frame,frame)
            if self._curveFilterNum > 0:
                self._curveFilter[self._curveFilterNum-1].apply(frame,frame)
            if self._enable_edge_detection:
                filters.strokeEdges(frame, frame)
            
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
        elif keycode == 99: #c
            self._channel_mixing_filter.filter_num += 1
        elif keycode == 102: #f
            if self._curveFilterNum < len(self._curveFilter):
                self._curveFilterNum += 1 
            else:
                self._curveFilterNum = 0
        elif keycode == 101: #e
            self._enable_edge_detection = not self._enable_edge_detection
        elif keycode == 100: #d
            self._enable_edge_detection = False
            self._curveFilterNum = 0
            self._channel_mixing_filter.filter_num = 0
        elif keycode == 27: #escape
            self._windowManager.destroyWindow()

if __name__=="__main__":
    Cameo().run()





