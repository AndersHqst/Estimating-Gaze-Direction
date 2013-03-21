import numpy as np
import cv
import cv2
import os

class EyeVideoLoader:
    

    def __init__(self):
        self.inputDirectory = os.path.normpath("C:/Users/David/Google Drev/edu 2.0/Machine Learning/project/videos/")
        self.outputDirectory = os.path.normpath("C:/Users/David/Downloads/")
        self.imageSize = (80,60)
        self.resultFile = os.path.join(self.outputDirectory, "eyeVideo.avi")

    

    def processEyeVideos(self):
        videoWriter = cv2.VideoWriter(self.resultFile, cv.CV_FOURCC('D','I','V','3'), 25.0, (self.imageSize[0], self.imageSize[1]), False)
        files = os.listdir(self.inputDirectory)
        for file in files:
            path = os.path.join(self.inputDirectory, file)
            video = cv2.VideoCapture(path)
            running, image = video.read()
            print file
            while (running):
                image = self.processEyeImage(image)
                videoWriter.write(image)
                running, image = video.read()
        videoWriter.release()

        print "All done!"
        raw_input()



    def processEyeImage(self, image):
        image = np.copy(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, self.imageSize)
        return image


    
    def loadImagesFromVideo(self, videoPath):
        video = cv2.VideoCapture(videoPath)
        images = None

        while (True):
            running, image = video.read()
            if not running:
                break

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.pyrDown(image)
            image = np.array(image)
            image = image.reshape(1, -1) # row vector

            if images is None:
                images = image
            else:
                images = np.concatenate((images, image), axis = 0)
        
        return images
