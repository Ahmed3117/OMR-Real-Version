import cv2
import numpy as np
from imutils.perspective import four_point_transform
from PIL import Image
import os
from datetime import datetime
from django.shortcuts import render
from .models import Exam,Students,StudentsScores

   
questionCount = 40
bubbleCount = 5
sqrAvrArea = 0
bubbleWidthAvr = 0
bubbleHeightAvr = 0
ovalCount = questionCount * bubbleCount
ANSWER_KEY = {
    0: 1, 1: 4,
    2: 2, 3: 0,
    4: 1, 5: 3,
    6: 2, 7: 4,
    8: 0, 9: 3,
    10: 1, 11: 1,
    12: 4, 13: 0,
    14: 3, 15: 1,
    16: 1, 17: 4,
    18: 0, 19: 3,
    20: 1, 21: 4,
    22: 2, 23: 0,
    24: 1, 25: 3,
    26: 2, 27: 4,
    28: 0, 29: 3,
    30: 1, 31: 1,
    32: 4, 33: 0,
    34: 3, 35: 1,
    36: 1, 37: 4,
    38: 0, 39: 3,
    40: 1,
}

    
def getCannyFrame( frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
    frame = cv2.Canny(gray, 127, 255)
    return frame

def getAdaptiveThresh( frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    adaptiveFrame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 7)
    # adaptiveFrame = canny = cv2.Canny(frame, 127, 255)
    return adaptiveFrame

def getFourPoints( canny):
    squareContours = []
    contours, hie = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        fourPoints = []
        i = 0
        for cnt in contours:

            (x, y), (MA, ma), angle = cv2.minAreaRect(cnt)

            epsilon = 0.04 * cv2.arcLength(cnt, False)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if len(approx) == 4 and aspect_ratio >= 0.9 and aspect_ratio <= 1.1:
                M = cv2.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                fourPoints.append((cx, cy))
                squareContours.append(cnt)
                i += 1
        squareContours = sorted(squareContours, key=cv2.contourArea, reverse=True)
        return fourPoints, squareContours

# We are using warping process for creative purposes
def getWarpedFrame( cannyFrame, frame):
    global sqrAvrArea
    fourPoints = np.array(getFourPoints(cannyFrame)[0], dtype="float32")
    fourContours = getFourPoints(cannyFrame)[1]

    if len(fourPoints) >= 4:
        newFourPoints = []
        newFourPoints.append(fourPoints[0])
        newFourPoints.append(fourPoints[1])
        newFourPoints.append(fourPoints[len(fourPoints) - 2])
        newFourPoints.append(fourPoints[len(fourPoints) - 1])

        newSquareContours = []
        newSquareContours.append(fourContours[0])
        newSquareContours.append(fourContours[1])
        newSquareContours.append(fourContours[len(fourContours) - 2])
        newSquareContours.append(fourContours[len(fourContours) - 1])

        for cnt in newSquareContours:
            area = cv2.contourArea(cnt)
            sqrAvrArea += area

        sqrAvrArea = int(sqrAvrArea / 4)

        newFourPoints = np.array(newFourPoints, dtype="float32")

        return four_point_transform(frame, newFourPoints)
    else:
        return None


def getOvalContours( adaptiveFrame):
    global bubbleWidthAvr
    global bubbleHeightAvr
    contours, hierarchy = cv2.findContours(adaptiveFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ovalContours = []

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0, True)
        ret = 0
        x, y, w, h = cv2.boundingRect(contour)

        # eliminating not ovals by approx lenght
        if (len(approx) > 15 and w / h <= 1.2 and w / h >= 0.8):

            mask = np.zeros(adaptiveFrame.shape, dtype="uint8")
            cv2.drawContours(mask, [contour], -1, 255, -1)

            ret = cv2.matchShapes(mask, contour, 1, 0.0)

            if (ret < 1):
                ovalContours.append(contour)
                bubbleWidthAvr += w
                bubbleHeightAvr += h
    bubbleWidthAvr = bubbleWidthAvr / len(ovalContours)
    bubbleHeightAvr = bubbleHeightAvr / len(ovalContours)
    firstovalContours = ovalContours        
    return firstovalContours

def x_cord_contour( ovalContour):
    global bubbleHeightAvr
    x, y, w, h = cv2.boundingRect(ovalContour)

    return y + x * bubbleHeightAvr

def y_cord_contour( ovalContour):
    global bubbleWidthAvr
    x, y, w, h = cv2.boundingRect(ovalContour)

    return x + y * bubbleWidthAvr
###########################################################################################################################################################


upperquestionCount = 6
upperbubbleCount = 10
uppersqrAvrArea = 0
upperbubbleWidthAvr = 0
upperbubbleHeightAvr = 0
upperovalCount = upperquestionCount * upperbubbleCount

def uppergetCannyFrame( frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
    frame = cv2.Canny(gray, 127, 255)
    return frame

def uppergetAdaptiveThresh( frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    adaptiveFrame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 7)
    # adaptiveFrame = canny = cv2.Canny(frame, 127, 255)
    return adaptiveFrame

def uppergetFourPoints( canny):
    squareContours = []
    contours, hie = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        fourPoints = []
        i = 0
        for cnt in contours:

            (x, y), (MA, ma), angle = cv2.minAreaRect(cnt)

            epsilon = 0.04 * cv2.arcLength(cnt, False)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if len(approx) == 4 and aspect_ratio >= 0.9 and aspect_ratio <= 1.1:
                M = cv2.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                fourPoints.append((cx, cy))
                squareContours.append(cnt)
                i += 1
        squareContours = sorted(squareContours, key=cv2.contourArea, reverse=True)
        return fourPoints, squareContours

# We are using warping process for creative purposes
def uppergetWarpedFrame( cannyFrame, frame):
    global uppersqrAvrArea
    fourPoints = np.array(uppergetFourPoints(cannyFrame)[0], dtype="float32")
    fourContours = uppergetFourPoints(cannyFrame)[1]

    if len(fourPoints) >= 4:
        newFourPoints = []
        newFourPoints.append(fourPoints[0])
        newFourPoints.append(fourPoints[1])
        newFourPoints.append(fourPoints[len(fourPoints) - 2])
        newFourPoints.append(fourPoints[len(fourPoints) - 1])

        newSquareContours = []
        newSquareContours.append(fourContours[0])
        newSquareContours.append(fourContours[1])
        newSquareContours.append(fourContours[len(fourContours) - 2])
        newSquareContours.append(fourContours[len(fourContours) - 1])

        for cnt in newSquareContours:
            area = cv2.contourArea(cnt)
            uppersqrAvrArea += area

        uppersqrAvrArea = int(uppersqrAvrArea / 4)

        newFourPoints = np.array(newFourPoints, dtype="float32")

        return four_point_transform(frame, newFourPoints)
    else:
        return None

def uppergetOvalContours( adaptiveFrame):
    global upperbubbleWidthAvr
    global upperbubbleHeightAvr
    contours, hierarchy = cv2.findContours(adaptiveFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ovalContours = []

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0, True)
        ret = 0
        x, y, w, h = cv2.boundingRect(contour)

        # eliminating not ovals by approx lenght
        if (len(approx) > 15 and w / h <= 1.2 and w / h >= 0.8):

            mask = np.zeros(adaptiveFrame.shape, dtype="uint8")
            cv2.drawContours(mask, [contour], -1, 255, -1)

            ret = cv2.matchShapes(mask, contour, 1, 0.0)

            if (ret < 1):
                ovalContours.append(contour)
                upperbubbleWidthAvr += w
                upperbubbleHeightAvr += h
    upperbubbleWidthAvr = upperbubbleWidthAvr / len(ovalContours)
    upperbubbleHeightAvr = upperbubbleHeightAvr / len(ovalContours)
    upperovalContours = ovalContours[:upperovalCount]
    
    return upperovalContours

def upper_x_cord_contour( ovalContour):
    x, y, w, h = cv2.boundingRect(ovalContour)

    return y + x * upperbubbleHeightAvr

def upper_y_cord_contour( ovalContour):
    x, y, w, h = cv2.boundingRect(ovalContour)

    return x + y * upperbubbleWidthAvr
#------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------
def printscore(request):
    score = 0
    
    global ovalCount
    global bubbleCount
    # Replace "path/to/folder" with the actual path to the folder containing your images
    folder_path = "C:/Users/DOLPHIN-ARTS/Desktop/ahmed issa/images"
    # Define the name of the folder you want to create
    folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Create the folder 
    new_folder_path =folder_path+"/"+folder_name

    os.makedirs(new_folder_path)
    # new_folder_path = new_folder_path.replace("\\", "/")
    print(new_folder_path)
    score = 0
    secretenumber = ''
    # Loop through all the files in the folder
    for filename in os.listdir(folder_path):
        # print(folder_path+"/"+filename)
    
        # Check if the file is an image file (you may need to modify this to match your file extensions)
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".PNG") or filename.endswith(".gif"):
            # Load the input image
            input_image = Image.open(folder_path+"/"+filename)

            # Get the dimensions of the input image
            width, height = input_image.size

            # Calculate the coordinates to split the image in half
            split_coord = height // 4

            # Split the image into upper and lower halves
            upper_half = input_image.crop((0, 0, width, split_coord))
            lower_half = input_image.crop((0, split_coord, width, height))
            # Convert the images to RGB mode
            upper_half = upper_half.convert('RGB')
            lower_half = lower_half.convert('RGB')

            # Save the two halves as separate files
            upper_half.save(new_folder_path+"/upper_"+filename[:-4]+".jpg")
            lower_half.save(new_folder_path+"/lower_"+filename[:-4]+".jpg")
            #-------------------------------------------------------------------------------------------------
            image = cv2.imread(new_folder_path+"/lower_"+filename[:-4]+".jpg")
            h = int(round(600 * image.shape[0] / image.shape[1]))
            frame = cv2.resize(image, (600, h), interpolation=cv2.INTER_LANCZOS4)
            cannyFrame = getCannyFrame(frame)
            warpedFrame = getWarpedFrame(cannyFrame, frame)
            adaptiveFrame = getAdaptiveThresh(warpedFrame)

            ovalContours = getOvalContours(adaptiveFrame)
            print("lower----------------")
            print(len(ovalContours))
            print(ovalCount)
            if (len(ovalContours) == ovalCount):

                ovalContours = sorted(ovalContours, key=y_cord_contour, reverse=False)
                
                for (q, i) in enumerate(np.arange(0, len(ovalContours), bubbleCount)):
                    bubbles = sorted(ovalContours[i:i + bubbleCount], key=x_cord_contour,
                                    reverse=False)
                    for (j, c) in enumerate(bubbles):
                        area = cv2.contourArea(c)

                        mask = np.zeros(adaptiveFrame.shape, dtype="uint8")
                        cv2.drawContours(mask, [c], -1, 255, -1)
                        mask = cv2.bitwise_and(adaptiveFrame, adaptiveFrame, mask=mask)
                        total = cv2.countNonZero(mask)
                        answer = ANSWER_KEY[q]

                        x, y, w, h = cv2.boundingRect(c)

                        isBubbleSigned = ((float)(total) / (float)(area)) > 1
                        if (isBubbleSigned):
                            if (answer == j):
                                # And calculate score
                                score += 1
                print(score)
            #----------------------------------------------------------------------------------------
            upperimage = cv2.imread(new_folder_path+"/upper_"+filename[:-4]+".jpg")
            upper_h = int(round(600 * upperimage.shape[0] / upperimage.shape[1]))
            upperframe = cv2.resize(upperimage, (600, upper_h), interpolation=cv2.INTER_LANCZOS4)
            uppercannyFrame = uppergetCannyFrame(upperframe)
            upperwarpedFrame = uppergetWarpedFrame(uppercannyFrame, upperframe)
            upperadaptiveFrame = uppergetAdaptiveThresh(upperwarpedFrame)
            

            upperovalContours = uppergetOvalContours(upperadaptiveFrame)
            print("upper----------------")
            print(len(upperovalContours))
            print(upperovalCount)
            if (len(upperovalContours) == upperovalCount):

                upperovalContours = sorted(upperovalContours, key=upper_y_cord_contour, reverse=False)
                
                
                for (q, i) in enumerate(np.arange(0, len(upperovalContours), upperbubbleCount)):

                    upperbubbles = sorted(upperovalContours[i:i + upperbubbleCount], key=upper_x_cord_contour,
                                    reverse=False)

                    for (j, c) in enumerate(upperbubbles):
                        area = cv2.contourArea(c)

                        uppermask = np.zeros(upperadaptiveFrame.shape, dtype="uint8")
                        cv2.drawContours(uppermask, [c], -1, 255, -1)
                        uppermask = cv2.bitwise_and(upperadaptiveFrame, upperadaptiveFrame, mask=uppermask)
                        total = cv2.countNonZero(uppermask)
                        x, y, w, h = cv2.boundingRect(c)

                        upperisBubbleSigned = ((float)(total) / (float)(area)) > 1
                        if (upperisBubbleSigned):
                            secretenumber += str(j)
                print(secretenumber)

                student = Students.objects.get(military_number=secretenumber)
                exam = Exam.objects.get(id=1)
                student_score = StudentsScores.objects.update_or_create(exam=exam,student=student,score=score)
        secretenumber=''

    context = {'score':score}
    return render(request,'scanner/main.html' ,context)