# Cameo

This app takes real time video stream from webcam and applies different processing techniques, which can be selected using the keyboard.

## Usage
1- Clone repo and go to directory
2- Execute ```python cameo.py```

Keyboard functions:
  * space: take screenshot. Saved as /screenshot.png
  * tab: start/stop video recording. Saved as /screencast.avi
  * x: start/stop bounding box drawing for real time face tracking
  * c: switch between channel-mixing filter
  * f: switch between convolutional filters
  * e: enable/disable edge detection
  * esc: exit app

## Current features:
* Real-time face tracking and swapping:
    * Uses Haar Cascades to track face, eyes and nose and swaps faces when there two or more faces on the screen
* Dynamic filtering:
    * Different filters available, based on techniques such as channel-mixing, curve-based filters and 2D convolution

## Upcoming Features
* Support for video loading from file (.mp4)
* Landmark detection for hand tracking using MediaPipe
