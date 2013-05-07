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
        self.resetLoadedData()

    
    def resetLoadedData(self):
        self.data = []
        self.targets = []
        self.people = []
        self.personIds = {}
        self.nextPersonId = 0
        self.singleFeature = []

    def normalizeSampleImages(self):
        paths = [
                  os.path.normpath("C:/Users/David/Downloads/sampleimages/sample1.png"),
                  os.path.normpath("C:/Users/David/Downloads/sampleimages/sample2.png"),
                  os.path.normpath("C:/Users/David/Downloads/sampleimages/sample3.png"),
                  os.path.normpath("C:/Users/David/Downloads/sampleimages/sample4.png")
                 ]
        for path in paths:
            image = cv2.imread(path)
            self.normalizeImage(image)
    

    def resizeEyeVideos(self):
        cv2.namedWindow("Debug")

        files = os.listdir(self.inputDirectory)
        #random.shuffle(files)
        #files = ["E 4.mp4"]

        for file in files:
            self.resizeEyeVideo(file)

        print "All done!"
        raw_input()


    def loadSingleFeatureData(self):
        files = os.listdir(self.inputDirectory)

        for file in files:
            self.loadSingleFeatureDataFrom(file)

        return self.singleFeature


    def loadSingleFeatureDataFrom(self, fileName):
        print fileName
        inputPath = os.path.join(self.inputDirectory, fileName)

        videoReader = cv2.VideoCapture(inputPath)
        running, image = videoReader.read()

        while (running):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            feature = normalizer.extractSingleFeature(image)
            if feature is not None:
                self.singleFeature.append(feature)

            running, image = videoReader.read()





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

        cv2.namedWindow("Debug")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = normalizer.normalizeImage(image)

        return image

    
    def loadDataFromVideos(self):
        self.resetLoadedData()

        files = os.listdir(self.resizedVideoDirectory)
        files = filter(lambda file: file.find(".avi") != -1, files)

        for file in files:
            self.loadDataFromVideo(file)

        data = np.array(self.data).reshape((-1, 42*28))
        targets = np.array(self.targets)
        people = np.array(self.people)

        return data, targets, people

    def getPersonId(self, name):
        if not (name in self.personIds):
            self.personIds[name] = self.nextPersonId
            self.nextPersonId += 1
        
        return self.personIds[name]


    def loadDataFromVideo(self, fileName):
        print fileName

        videoPath = os.path.join(self.resizedVideoDirectory, fileName)
        video = cv2.VideoCapture(videoPath)
        labels = fileName.split('.')[0].split(' ')
        name = labels[0]
        personId = self.getPersonId(name)
        target = int(labels[1])

        while (True):
            running, image = video.read()
            if not running:
                break

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (42,28))

            self.data.append(image)
            self.targets.append(target)
            self.people.append(personId)

