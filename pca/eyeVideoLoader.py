import numpy as np
import cv
import cv2
import os
import normalizer
import random

class EyeVideoLoader:

    def __init__(self):
        self.inputDirectory = os.path.normpath("C:/Users/David/Google Drev/edu 2.0/Machine Learning/project/videos")
        self.resizedVideoDirectory = os.path.normpath("C:/Users/David/Google Drev/edu 2.0/Machine Learning/project/videos/normalized (2)")
        #self.resizedVideoDirectory = os.path.normpath("C:/Users/David/Downloads/pit/output")
        #self.inputDirectory = os.path.normpath("/Users/ahkj/Google Drev/Machine Learning/project/videos")
        #self.resizedVideoDirectory = os.path.normpath("/Users/ahkj/Google Drev/Machine Learning/project/videos/normalized (2)")
        self.imageSize = (360,240)
        self.data = []
        self.targets = []
    

    def resizeEyeVideos(self):
        cv2.namedWindow("Threshold")
        cv2.namedWindow("Debug")

        files = os.listdir(self.inputDirectory)
        #random.shuffle(files)
        #files = ["E 4.mp4"]

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
        skippedFrames = 0
        while (running):
            # image = self.resizeEyeImage(image)
            image = self.normalizeImage(image)
            if image is not None:
                videoWriter.write(image)
            else:
                skippedFrames += 1
            running, image = videoReader.read()

        videoWriter.release()
        print "Skipped frames:", skippedFrames


    def normalizeImage(self, image):
        image = np.copy(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = normalizer.normalizeImage(image)

        return image

    
    def loadDataFromVideos(self):
        self.data = []
        self.targets = []

        files = os.listdir(self.resizedVideoDirectory)
        files = filter(lambda file: file.find(".avi") != -1, files)

        for file in files:
            self.loadDataFromVideo(file)

        data = np.array(self.data).reshape((-1, 42*28))
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
            image = cv2.resize(image, (42,28))

            #average = np.average(image)
            mean = np.mean(image)
            #alpha = 128 / average
            beta = 128 - mean
            image = cv2.convertScaleAbs(image, beta = beta)
            #image = cv2.equalizeHist(image)

            self.data.append(image)
            self.targets.append(target)

