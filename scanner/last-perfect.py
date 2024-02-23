import cv2
import numpy as np
from imutils.perspective import four_point_transform
from PIL import Image
import os
from datetime import datetime
from django.shortcuts import render,redirect
from .models import Exam,Students,StudentsScores

#############################################################
# C:\Users\nozom\Desktop\nnn
# home front not modified
# move analysis to models 

############################################################

sqrAvrArea = 0
bubbleWidthAvr = 0
bubbleHeightAvr = 0

def getCannyFrame( frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
    frame = cv2.Canny(gray, 127, 255)
    return frame

def getAdaptiveThresh( frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    adaptiveFrame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 1)
    # adaptiveFrame = cv2.Canny(frame, 127, 255)
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
            if len(approx) == 4 and aspect_ratio >= 0.7 and aspect_ratio <= 1.2:
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
        if (len(approx) > 10 and w / h <= 1.2 and w / h >= 0.8):

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


# We are using warping process for creative purposes
def uppergetWarpedFrame( cannyFrame, frame):
    global uppersqrAvrArea
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
        if (len(approx) > 10 and w / h <= 1.2 and w / h >= 0.8):

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
failed_images = []
#-------------------------------------------------------------------------------------------------
import os
from datetime import datetime
from django.conf import settings
def printscore(request):
    exam_title = request.POST.get("examtitle")
    examiner = request.POST.get("examiner")
    questionCount = int(request.POST.get("questionsnumber"))
    bubbleCount = int(request.POST.get("answeres_number"))
    ovalCount = questionCount * bubbleCount
    print("tttttttttttttttttttttttttttttttttttt")
    print(questionCount)
    print(bubbleCount)
    print(ovalCount)
    print("tttttttttttttttttttttttttttttttttttt")
    question_score = int(request.POST.get("onequestionscore"))
    folder_path = request.POST.get("folderpath")

    exam = Exam.objects.create(title=exam_title, examiner=examiner, questions_number=questionCount, answeres_number=bubbleCount, question_score=question_score, folder_path=folder_path)
    exam.save()

    ANSWER_KEY1 = {}
    ANSWER_KEY2 = {}

    for box in range(0,25):
        ans=request.POST.get("box-"+str(box)+"-radio")
        if ans == None :
            ANSWER_KEY1[box] = ans
        else:
            ANSWER_KEY1[box] = int(ans)
    print(ANSWER_KEY1)
    for box in range(25,questionCount):
        ans=request.POST.get("box-"+str(box)+"-radio")
        if ans == None :
            ANSWER_KEY2[box-25] = ans
        else:
            ANSWER_KEY2[box-25] = int(ans)
    print(ANSWER_KEY2)


    # ANSWER_KEY1 = {0: 0, 1: 0, 2: 0, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None, 9: None, 10: None, 11: None, 12: None, 13: None, 14: None, 15: None, 16: None, 17: None, 18: None, 19: None, 20: None, 21: None, 22: None, 23: None, 24: None}
    # ANSWER_KEY2= {0: 0, 1: 0, 2: 0, 3: 0, 4: None, 5: None, 6: None, 7: None, 8: None, 9: None, 10: None, 11: None, 12: None, 13: None, 14: None, 15: None, 16: None, 17: None, 18: None, 19: None, 20: None, 21: None, 22: None, 23: None, 24: None}
    BubbleSigned=0
    score = 0
    secretenumber = ''
    global failed_images
    failed_images = []
    # Replace "path/to/folder" with the actual path to the folder containing your images
    # folder_path = "C:/Users/DOLPHIN-ARTS/Desktop/ahmed issa/images"
    # Define the name of the folder you want to create
    # Generate folder name with current date and time
    folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Define the folder path on the server
    new_folder_path = os.path.join(settings.MEDIA_ROOT, folder_name)

    # Create the folder if it doesn't exist
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    # for filename in os.listdir(folder_path):
    #     if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".PNG") :
    #         img = Image.open(folder_path+"/"+filename)
    #         threshold = 198
    #         img = img.point(lambda x: 0 if x < threshold else 255)
    #         img = img.convert("1")
    #         img.save(folder_path+"/"+filename)

    # Loop through all the files in the folder
    for filename in os.listdir(folder_path):
        # print(folder_path+"/"+filename)
        
    # Check if the file is an image file (you may need to modify this to match your file extensions)
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".PNG") :
            try:
                print(filename)
                # Load the input image
                input_image = Image.open(folder_path+"/"+filename)
                
                input_image = input_image.resize((794, 1123), resample=Image.BICUBIC)
                # Get the dimensions of the input image
                width, height = input_image.size
                

                # Calculate the coordinates to split the image in half
                split_coord = height // 3.8
                
                # Split the image into upper and lower halves
                upper_half = input_image.crop((0, 0, width, split_coord))
                lower_half = input_image.crop((0, split_coord, width, height))
                # Convert the images to RGB mode
                upper_half = upper_half.convert('RGB')
                lower_half = lower_half.convert('RGB')

                # Save the two halves as separate files
                upper_half.save(new_folder_path+"/upper_"+filename[:-4]+".jpg")
                lower_half.save(new_folder_path+"/lower_"+filename[:-4]+".jpg")

                #----------------------------------------------------------------------------------------------
                
                lower_input_image = Image.open(new_folder_path+"/lower_"+filename[:-4]+".jpg")
                # Get the dimensions of the input image
                width, height = lower_input_image.size
                # Calculate the coordinates to split the image in half
                split_coord = width // 2

                # Split the image into upper and lower halves
                left_half = lower_input_image.crop((0, 0, split_coord, height))
                right_half = lower_input_image.crop((split_coord, 0, width, height))
                
                

                # Save the two halves as separate files
                left_half.save(new_folder_path+"/left_"+filename[:-4]+".jpg")
                
                right_half.save(new_folder_path+"/right_"+filename[:-4]+".jpg")




                dimentions = [400,300,280,310,250,290,200,600]
                #-------------------------------------------------------------------------------------------------
                
                left_image = cv2.imread(new_folder_path+"/left_"+filename[:-4]+".jpg")
                opened_left_image = Image.open(new_folder_path+"/left_"+filename[:-4]+".jpg")
                w, h = opened_left_image.size
                frame = cv2.resize(left_image, (w, h), interpolation=cv2.INTER_LANCZOS4)
                cannyFrame = getCannyFrame(frame)
                warpedFrame = getWarpedFrame(cannyFrame, frame)
                adaptiveFrame = getAdaptiveThresh(warpedFrame)
                ovalContours = getOvalContours(adaptiveFrame)
                
                print(len(ovalContours))
                if (len(ovalContours) == ovalCount/2):
                    print("left----------------" )
                    print(len(ovalContours))
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
                            answer = ANSWER_KEY1[q]
                            
                            
                            x, y, w, h = cv2.boundingRect(c)

                            isBubbleSigned = ((float)(total) / (float)(area)) > 1
                            if (isBubbleSigned):
                                BubbleSigned += 1
                                if (answer == j):

                                    # And calculate score
                                    score += 1
                else:
                    for dim in dimentions:
                        left_image = cv2.imread(new_folder_path+"/left_"+filename[:-4]+".jpg")
                        h = int(round(dim * left_image.shape[0] / left_image.shape[1]))
                        frame = cv2.resize(left_image, (dim, h), interpolation=cv2.INTER_LANCZOS4)
                        cannyFrame = getCannyFrame(frame)
                        warpedFrame = getWarpedFrame(cannyFrame, frame)
                        adaptiveFrame = getAdaptiveThresh(warpedFrame)
                        ovalContours = getOvalContours(adaptiveFrame)
                        print(len(ovalContours))
                        if (len(ovalContours) == ovalCount/2):
                            print("left----------------" )
                            print(len(ovalContours))
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
                                    answer = ANSWER_KEY1[q]

                                    x, y, w, h = cv2.boundingRect(c)

                                    isBubbleSigned = ((float)(total) / (float)(area)) > 1
                                    if (isBubbleSigned):
                                        BubbleSigned += 1
                                        if (answer == j):
                                            # And calculate score
                                            score += 1
                            break
                        else:
                            if dimentions.index(dim) == len(dimentions)-1 :
                                print('error from left')
                                print(len(ovalContours))
                                if filename not in failed_images:
                                    failed_images.append(filename)

                right_image = cv2.imread(new_folder_path+"/right_"+filename[:-4]+".jpg")
                opened_right_image = Image.open(new_folder_path+"/right_"+filename[:-4]+".jpg")
                w, h = opened_right_image.size
                frame = cv2.resize(right_image, (w, h), interpolation=cv2.INTER_LANCZOS4)
                cannyFrame = getCannyFrame(frame)
                warpedFrame = getWarpedFrame(cannyFrame, frame)
                adaptiveFrame = getAdaptiveThresh(warpedFrame)
                ovalContours = getOvalContours(adaptiveFrame)
                # h = int(round(320 * warpedFramee.shape[0] / warpedFramee.shape[1]))
                # frame = cv2.resize(right_image, (300, h), interpolation=cv2.INTER_LANCZOS4)

                # cannyFrame = getCannyFrame(frame)
                # warpedFrame = getWarpedFrame(cannyFrame, frame)
                # adaptiveFrame = getAdaptiveThresh(warpedFrame)
                # ovalContours = getOvalContours(adaptiveFrame)
                
                # print('ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo')
                # print(len(ovalContours))
                # print(ovalCount)
                # print('ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo')
                if (len(ovalContours) == ovalCount/2):
                    print("right----------------")
                    print(len(ovalContours))
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
                            answer = ANSWER_KEY2[q]
                            
                            x, y, w, h = cv2.boundingRect(c)

                            isBubbleSigned = ((float)(total) / (float)(area)) > 1
                            if (isBubbleSigned):
                                BubbleSigned += 1
                                if (answer == j):
                                    
                                    # And calculate score
                                    score += 1
                    
                    
                else:
                    for dim in dimentions:
                        right_image = cv2.imread(new_folder_path+"/right_"+filename[:-4]+".jpg")
                        h = int(round(dim * right_image.shape[0] / right_image.shape[1]))
                        frame = cv2.resize(right_image, (dim, h), interpolation=cv2.INTER_LANCZOS4)
                        cannyFrame = getCannyFrame(frame)
                        warpedFrame = getWarpedFrame(cannyFrame, frame)
                        adaptiveFrame = getAdaptiveThresh(warpedFrame)
                        ovalContours = getOvalContours(adaptiveFrame)
                        if (len(ovalContours) == ovalCount/2):
                            print("left----------------" )
                            print(len(ovalContours))
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
                                    answer = ANSWER_KEY1[q]

                                    x, y, w, h = cv2.boundingRect(c)

                                    isBubbleSigned = ((float)(total) / (float)(area)) > 1
                                    if (isBubbleSigned):
                                        BubbleSigned += 1
                                        if (answer == j):
                                            # And calculate score
                                            score += 1
                            break
                        else:
                            if dimentions.index(dim) == len(dimentions)-1 :
                                print('error from right')
                                print(len(ovalContours))
                                if filename not in failed_images:
                                    failed_images.append(filename) 
                #----------------------------------------------------------------------------------------
                
                # print('nnnnnnnnnnnnnnnnnn')
                # print(BubbleSigned)
                # if BubbleSigned > 50 :
                #     score = score - (BubbleSigned-50)


                upperimage = cv2.imread(new_folder_path+"/upper_"+filename[:-4]+".jpg")
                opened_upperimage = Image.open(new_folder_path+"/upper_"+filename[:-4]+".jpg")
                w, h = opened_upperimage.size
                frame = cv2.resize(upperimage, (w, h), interpolation=cv2.INTER_LANCZOS4)
                # h = int(round(600 * upperimage.shape[0] / upperimage.shape[1]))
                # frame = cv2.resize(upperimage, (600, h), interpolation=cv2.INTER_LANCZOS4)
                uppercannyFrame = getCannyFrame(frame)
                upperwarpedFrame = uppergetWarpedFrame(uppercannyFrame, frame)
                upperadaptiveFrame = getAdaptiveThresh(upperwarpedFrame)
                upperovalContours = uppergetOvalContours(upperadaptiveFrame)
                if (len(upperovalContours) == upperovalCount):
                    print("upper----------------")
                    print(len(upperovalContours))
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
                    try:
                        student = Students.objects.get(military_number=secretenumber)
                    except:
                    #     if len(secretenumber) < 6 :
                    #         student = Students.objects.create(military_number=secretenumber,student_name = filename)
                    #     else :
                        student = Students.objects.create(military_number=secretenumber,student_name = secretenumber)
                    #     student.save()
                    if filename not in failed_images:
                        # print('rrrrrrrrrrrrrrrrrrrrrr')
                        # print(score)
                        # print(question_score)
                        # print(BubbleSigned)
                        # print('rrrrrrrrrrrrrrrrrrrrrr')
                        if score < 0 :
                            score = 0
                        student_score = StudentsScores.objects.update_or_create(exam=exam,student=student,score=score*question_score)
                else:
                    upperimage = cv2.imread(new_folder_path+"/upper_"+filename[:-4]+".jpg")
                    # opened_upperimage = Image.open(new_folder_path+"/upper_"+filename[:-4]+".jpg")
                    # w, h = opened_upperimage.size
                    # frame = cv2.resize(upperimage, (w, h), interpolation=cv2.INTER_LANCZOS4)
                    h = int(round(600 * upperimage.shape[0] / upperimage.shape[1]))
                    frame = cv2.resize(upperimage, (600, h), interpolation=cv2.INTER_LANCZOS4)
                    uppercannyFrame = getCannyFrame(frame)
                    upperwarpedFrame = uppergetWarpedFrame(uppercannyFrame, frame)
                    upperadaptiveFrame = getAdaptiveThresh(upperwarpedFrame)
                    upperovalContours = uppergetOvalContours(upperadaptiveFrame)
                    if (len(upperovalContours) == upperovalCount):
                        print("upper----------------")
                        print(len(upperovalContours)) 
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
                        try:
                            student = Students.objects.get(military_number=secretenumber)
                        except:
                        #     if len(secretenumber) < 6 :
                        #         student = Students.objects.create(military_number=secretenumber,student_name = filename)
                        #     else :
                            student = Students.objects.create(military_number=secretenumber,student_name = secretenumber)
                        #     student.save()
                        if filename not in failed_images:
                            # print('rrrrrrrrrrrrrrrrrrrrrr')
                            # print(score)
                            # print(question_score)
                            # print(BubbleSigned)
                            # print('rrrrrrrrrrrrrrrrrrrrrr')
                            if score < 0 :
                                score = 0
                            student_score = StudentsScores.objects.update_or_create(exam=exam,student=student,score=score*question_score)
                    else:
                        upperimage = cv2.imread(new_folder_path+"/upper_"+filename[:-4]+".jpg")
                        # opened_upperimage = Image.open(new_folder_path+"/upper_"+filename[:-4]+".jpg")
                        # w, h = opened_upperimage.size
                        # frame = cv2.resize(upperimage, (w, h), interpolation=cv2.INTER_LANCZOS4)
                        h = int(round(400 * upperimage.shape[0] / upperimage.shape[1]))
                        frame = cv2.resize(upperimage, (400, h), interpolation=cv2.INTER_LANCZOS4)
                        uppercannyFrame = getCannyFrame(frame)
                        upperwarpedFrame = uppergetWarpedFrame(uppercannyFrame, frame)
                        upperadaptiveFrame = getAdaptiveThresh(upperwarpedFrame)
                        upperovalContours = uppergetOvalContours(upperadaptiveFrame)
                        if (len(upperovalContours) == upperovalCount):
                            print("upper----------------")
                            print(len(upperovalContours)) 
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
                            try:
                                student = Students.objects.get(military_number=secretenumber)
                            except:
                            #     if len(secretenumber) < 6 :
                            #         student = Students.objects.create(military_number=secretenumber,student_name = filename)
                            #     else :
                                student = Students.objects.create(military_number=secretenumber,student_name = secretenumber)
                            #     student.save()
                            if filename not in failed_images:
                                # print('rrrrrrrrrrrrrrrrrrrrrr')
                                # print(score)
                                # print(question_score)
                                # print(BubbleSigned)
                                # print('rrrrrrrrrrrrrrrrrrrrrr')
                                if score < 0 :
                                    score = 0
                                student_score = StudentsScores.objects.update_or_create(exam=exam,student=student,score=score*question_score)
        
                     
                        else:
                            print('error from upper')
                            print(len(upperovalContours))
                            if filename not in failed_images:
                                failed_images.append(filename)
                            # try:
                            #     student = Students.objects.get(military_number=filename)
                            # except:
                            #     student = Students.objects.create(military_number=secretenumber,student_name = filename)
                            #     student.save()
                            # if filename not in failed_images:
                            #     student_score = StudentsScores.objects.update_or_create(exam=exam,student=student,score=score*question_score)
                            # if filename not in failed_images:
                            #     failed_images.append(filename)
            except:
                if filename not in failed_images:
                    print("bbbbbbbbbbbbbbbbbbbbbbb")
                    failed_images.append(filename)
        secretenumber=''
        score=0
        BubbleSigned = 0
        
    return redirect('scanner:examscores' ,exam.id)


def getCurrentExamDegrees(request,pk):
    global failed_images
    exam = Exam.objects.get(id=pk)
    students_scores = exam.studentsscores_set.all()
    print(failed_images)
    
    exams = Exam.objects.all()
    context = {
        "students_scores" : students_scores,
        "exams" : exams,
        "current_exam" : exam,
        "failed_images" : failed_images,

    }
    return render(request,'scanner/students_scores.html' ,context)

def ExamsFilter(request):
    
    students_scores =[]
    search_exam_value = request.POST.get("search-exam")
    student_name = request.POST.get("student-name")
    student_secretenumber = request.POST.get("student-secretenumber")
    succase = request.POST.get("succeedorfailed")
    exams = Exam.objects.all()
    if search_exam_value =="all":
        exam = None
        students_scores = StudentsScores.objects.all()   
        if succase == 'succeed' :
            students_scores = students_scores.filter(score__range=(25,50))
        if succase == 'failed' :
            students_scores = students_scores.filter(score__range=(0,24))
        if student_name :
            students_scores = students_scores.filter(student__student_name__icontains=student_name)
        if student_secretenumber :
            students_scores = students_scores.filter(student__military_number__icontains=student_secretenumber)
    else:
        exam = Exam.objects.get(id=search_exam_value)
        students_scores = exam.studentsscores_set.all()
        if succase == 'succeed' :
            students_scores = students_scores.filter(score__range=(25,50))
        if succase == 'failed' :
            students_scores = students_scores.filter(score__range=(0,24))
        if student_name :
            students_scores = students_scores.filter(student__student_name__icontains=student_name)
        if student_secretenumber :
            students_scores = students_scores.filter(student__military_number__icontains=str(student_secretenumber))
    
      
    context = {
        "students_scores" : students_scores,
        "exams" : exams,
        "current_exam" : exam,
    }
    return render(request,'scanner/students_scores.html' ,context)

def home(request):
    exams = Exam.objects.all()
    context = {
        "exams" : exams,
    }
    return render(request,'scanner/home.html',context)