import numpy as np
import cv
import cv2
import os
import normalizer


class EyeVideoLoader:
    

    def __init__(self):
        self.inputDirectory = os.path.normpath("C:/Users/David/Google Drev/edu 2.0/Machine Learning/project/videos")
        self.resizedVideoDirectory = os.path.normpath("C:/Users/David/Google Drev/edu 2.0/Machine Learning/project/videos/normalized (1)")
        self.imageSize = (80,60)
        self.data = []
        self.targets = []
    

    def resizeEyeVideos(self):
        files = os.listdir(self.inputDirectory)
        for file in files:
            self.resizeEyeVideo(file)

        print "All done!"
        raw_input()


    def resizeEyeVideo(self, fileName):
        print fileName
        inputPath = os.path.join(self.inputDirectory, fileName)
        outputPath = os.path.join(self.resizedVideoDirectory, fileName.replace(".mp4", ".avi"))
        
        videoWriter = cv2.VideoWriter(outputPath, cv.CV_FOURCC('D','I','V','3'), 25.0, self.imageSize, False)
        videoReader = cv2.VideoCapture(inputPath)

        running, image = videoReader.read()
        while (running):
            # image = self.resizeEyeImage(image)
            image = self.resizeAndNormalize(image)
            if image is not None:
                videoWriter.write(image)
            else:
                print "Skipping frame"
            running, image = videoReader.read()

        videoWriter.release()


    def resizeEyeImage(self, image):
        image = np.copy(image)
        image = cv2.resize(image, self.imageSize)
        return image


    def resizeAndNormalize(self, image):
        cv2.namedWindow("Threshold")
        image = np.copy(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = normalizer.normalizeImage(image)
        if image is None:
            return None
        else:
            return cv2.resize(image, self.imageSize)

    
    def loadDataFromVideos(self):
        self.data = []
        self.targets = []

        files = os.listdir(self.resizedVideoDirectory)
        files = filter(lambda file: file.find(".avi") != -1, files)

        for file in files:
            self.loadDataFromVideo(file)

        data = np.array(self.data).reshape((-1, 1200))
        targets = np.array(self.targets)

        return data, targets

    def loadDataFromVideo(self, fileName):
        print fileName

        videoPath = os.path.join(self.resizedVideoDirectory, fileName)
        video = cv2.VideoCapture(videoPath)
        target = int(fileName.split('.')[0].split(' ')[1])

        while (True):
            running, image = video.read()
            if not running:
                break

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.pyrDown(image)
            self.data.append(image)
            self.targets.append(target)

