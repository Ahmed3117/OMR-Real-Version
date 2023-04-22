import cv2
import numpy as np
from imutils.perspective import four_point_transform
from PIL import Image
import os
from datetime import datetime
from django.shortcuts import render,redirect
from .models import Exam,Students,StudentsScores
from PIL import Image
   

sqrAvrArea = 0
bubbleWidthAvr = 0
bubbleHeightAvr = 0

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
    exam_title = request.POST.get("examtitle")
    examiner = request.POST.get("examiner")
    questionCount = int(request.POST.get("questionsnumber"))
    bubbleCount = int(request.POST.get("answeres_number"))
    ovalCount = questionCount * bubbleCount
    # myfolderpath = request.POST.get("folderpath")
    folder_path = request.POST.get("folderpath")
    print(folder_path)
    print("###########################")
    # folder_path="C:/Users/DOLPHIN-ARTS/Desktop/ahmed issa/images"
    question_score = int(request.POST.get("onequestionscore"))
    # test = request.FILES["folderpath"]
    # print(test)
    # print("###########################")
    # print(examtitle)
    # print(examtitle)
    # print(examtitle)
    # print(folder_path)

    # create exam
    exam = Exam.objects.create(title =exam_title,examiner = examiner,questions_number=questionCount,answeres_number=bubbleCount,question_score=question_score,folder_path=folder_path)
    exam.save()

    # ANSWER_KEY = {
    # 0: 1, 1: 4,
    # 2: 2, 3: 0,

    # 4: 1, 5: 3,
    # 6: 2, 7: 4,

    # 8: 0, 9: 3,
    # 10: 1, 11: 1,

    # 12: 4, 13: 0,
    # 14: 3, 15: 1,

    # 16: 1, 17: 4,
    # 18: 0, 19: 3,

    # 22: 2, 23: 0,
    # 20: 1, 21: 4,

    # 24: 1, 25: 3,
    # 26: 2, 27: 4,

    # 28: 0, 29: 3,
    # 30: 1, 31: 1,

    # 32: 4, 33: 0,
    # 34: 3, 35: 1,

    # 36: 1, 37: 4,
    # 38: 0, 39: 3,

    # 40: 1,41: 1,
    # 42: 4, 43: 0,

    # 44: 4, 45: 1,
    # 46: 1, 47: 4,

    # 48: 0, 49: 3,
    # 50: 1,51: 1,
    # 52: 4, 53: 0,
    # 54: 5, 55: 1,
    # 56: 1, 57: 4,
    # 58: 0, 59: 3,
    # 60: 1,61: 1,
    # 62: 4, 63: 0,
    # 64: 4, 65: 1,
    # 66: 1, 67: 4,
    # 68: 0, 69: 3,
    # 70: 1,71: 1,
    # 72: 4, 73: 0,
    # 74: 5, 75: 1,
    # 76: 1, 77: 4,
    # 78: 0, 79: 3,
    # 80: 1,81: 1,
    # 82: 4, 83: 0,
    # 84: 5, 85: 1,
    # 86: 1, 87: 4,
    # 88: 0, 89: 3,
    # 90: 1,91: 1,
    # 92: 4, 93: 0,
    # 94: 5, 95: 1,
    # 96: 1, 97: 4,
    # 98: 0, 99: 3,
    # 100: 1,
    # }

    ANSWER_KEY = {}

    for box in range(0,questionCount):
        ans=request.POST.get("box-"+str(box)+"-radio")
        if ans == None :
            ANSWER_KEY[box] = ans
        else:
            ANSWER_KEY[box] = int(ans)
    print(ANSWER_KEY)


    score = 0
    secretenumber = ''
    # Replace "path/to/folder" with the actual path to the folder containing your images
    # folder_path = "C:/Users/DOLPHIN-ARTS/Desktop/ahmed issa/images"
    # Define the name of the folder you want to create
    folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Create the folder 
    new_folder_path =folder_path+"/"+folder_name

    os.makedirs(new_folder_path)
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".PNG") or filename.endswith(".gif"):
            img = Image.open(folder_path+"/"+filename)
            img = img.resize((794, 1123), resample=Image.BICUBIC)
            img.save(folder_path+"/"+filename)
    # Loop through all the files in the folder
    for filename in os.listdir(folder_path):
        # print(folder_path+"/"+filename)
        try:
        # Check if the file is an image file (you may need to modify this to match your file extensions)
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".PNG") or filename.endswith(".gif"):


                

               
                # img = Image.open(folder_path+"/"+filename)
                # img = img.resize((794, 1123), resample=Image.BICUBIC)
                # img.save(folder_path+"/"+filename)
    


                # Load the input image
                input_image = Image.open(folder_path+"/"+filename)

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





                #-------------------------------------------------------------------------------------------------
                left_image = cv2.imread(new_folder_path+"/left_"+filename[:-4]+".jpg")
                h = int(round(320 * left_image.shape[0] / left_image.shape[1]))
                frame = cv2.resize(left_image, (320, h), interpolation=cv2.INTER_LANCZOS4)
                cannyFrame = getCannyFrame(frame)
                warpedFrame = getWarpedFrame(cannyFrame, frame)
                adaptiveFrame = getAdaptiveThresh(warpedFrame)

                ovalContours = getOvalContours(adaptiveFrame)
                print("left----------------")
                print(len(ovalContours))
                
                if (len(ovalContours) == 125):

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
                    

                right_image = cv2.imread(new_folder_path+"/right_"+filename[:-4]+".jpg")
                # cannyFramee = getCannyFrame(right_image)
                # warpedFramee = getWarpedFrame(cannyFramee, right_image)
                h = int(round(320 * right_image.shape[0] / right_image.shape[1]))
                frame = cv2.resize(right_image, (320, h), interpolation=cv2.INTER_LANCZOS4)
                cannyFrame = getCannyFrame(frame)
                warpedFrame = getWarpedFrame(cannyFrame, frame)
                adaptiveFrame = getAdaptiveThresh(warpedFrame)

                ovalContours = getOvalContours(adaptiveFrame)
                print("right----------------")
                print(len(ovalContours))
                
                if (len(ovalContours) == 125):

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
                    try:
                        student = Students.objects.get(military_number=secretenumber)
                    except:
                        student = Students.objects.create(military_number=secretenumber,student_name = secretenumber)
                        student.save()
                    
                    student_score = StudentsScores.objects.update_or_create(exam=exam,student=student,score=score*question_score)
            secretenumber=''
            score=0
        except:
            print('invalid image')
    return redirect('scanner:examscores' ,exam.id)


def getCurrentExamDegrees(request,pk):
    exam = Exam.objects.get(id=pk)
    students_scores = exam.studentsscores_set.all()
    exams = Exam.objects.all()
    context = {
        "students_scores" : students_scores,
        "exams" : exams,
        "current_exam" : exam,
    }
    return render(request,'scanner/students_scores.html' ,context)

def ExamsFilter(request):
    exam = ''
    students_scores =[]
    search_exam_value = request.POST.get("search-exam")
    student_name = request.POST.get("student-name")
    student_secretenumber = request.POST.get("student-secretenumber")
    exams = Exam.objects.all()
    if search_exam_value =="all":
        students_scores = StudentsScores.objects.all()   
        if student_name :
            students_scores = students_scores.filter(student__student_name__icontains=student_name)
        if student_secretenumber :
            students_scores = students_scores.filter(student__military_number__icontains=student_secretenumber)
    else:
        exam = Exam.objects.get(id=search_exam_value)
        students_scores = exam.studentsscores_set.all()
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
    return render(request,'scanner/home.html')