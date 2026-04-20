# OpenCV-Project
**idea:** given an image, use reference images to best translate it from ASL to English 

The original program project.py has very low confidence and accuracy which is difficult to improve by only using templateMatch edge finding, as many hand gestures are very similar in edges even if they do not look similar. this results in high inaccuracy (lots of intended letters are registered as other ones).

Currently attempting to use google's mediapipe API in order to improve accuracy/hand detection and allow for detection of multiple hands (still low accuracy)

