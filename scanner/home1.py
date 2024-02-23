import cv2
import numpy as np
from imutils.perspective import four_point_transform
from PIL import Image
import os
from datetime import datetime
from django.shortcuts import render,redirect
from .models import Exam,Students,StudentsScores,Files
from django.conf import settings

#127.0.0.1
#3306
#root
#withALLAH

#############################################################
# C:\Users\nozom\Desktop\nnn
# home front not modified
# move analysis to models 

############################################################

sqrAvrArea = 0
bubbleWidthAvr = 0
bubbleHeightAvr = 0
upperquestionCount = 6
upperbubbleCount = 10
upperovalCount = upperquestionCount * upperbubbleCount


def getCannyFrame( frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
    frame = cv2.Canny(gray, 127, 255)
    return frame

def getAdaptiveThresh( frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    adaptiveFrame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 1)
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
    ovalContours = ovalContours        
    return ovalContours

def x_cord_contour( ovalContour):
    global bubbleHeightAvr
    x, y, w, h = cv2.boundingRect(ovalContour)

    return y + x * bubbleHeightAvr

def y_cord_contour( ovalContour):
    global bubbleWidthAvr
    x, y, w, h = cv2.boundingRect(ovalContour)

    return x + y * bubbleWidthAvr
###########################################################################################################################################################


def printscore(request):
    score = 0
    secretenumber = ''
    global failed_images
    failed_images = []
    exam_title = request.POST.get("examtitle")
    examiner = request.POST.get("examiner")
    questionCount = int(request.POST.get("questionsnumber"))
    bubbleCount = int(request.POST.get("answeres_number"))
    ovalCount = questionCount * bubbleCount
    question_score = int(request.POST.get("onequestionscore"))
    files = request.FILES.getlist('files')
    exam = Exam.objects.create(title=exam_title, examiner=examiner, questions_number=questionCount, answeres_number=bubbleCount, question_score=question_score)
    exam.save()
    for f in files:
        file_obj = Files.objects.create(file=f,exam=exam)
        file_obj.save()
    uploaded_files_path = os.path.dirname(file_obj.file.path)
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
    folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  
    new_folder_path = os.path.join(uploaded_files_path, folder_name)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    # for filename in os.listdir(uploaded_files_path):
    #     if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".PNG") :
    #         img = Image.open(uploaded_files_path+"/"+filename)
    #         threshold = 198
    #         img = img.point(lambda x: 0 if x < threshold else 255)
    #         img = img.convert("1")
    #         img.save(uploaded_files_path+"/"+filename)
    for filename in os.listdir(uploaded_files_path): 
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".PNG") :
            try:
                print(filename)
                # Load the input image
                input_image = Image.open(uploaded_files_path+"/"+filename)
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

                if (len(ovalContours) == ovalCount/2):
                    ovalContours = sorted(ovalContours, key=y_cord_contour, reverse=False)
                    for (q, i) in enumerate(np.arange(0, len(ovalContours), bubbleCount)):
                        bubbles = sorted(ovalContours[i:i + bubbleCount], key=x_cord_contour,reverse=False)
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
                if (len(ovalContours) == ovalCount/2):
                    print("right----------------")
                    print(len(ovalContours))
                    ovalContours = sorted(ovalContours, key=y_cord_contour, reverse=False)
                    for (q, i) in enumerate(np.arange(0, len(ovalContours), bubbleCount)):
                        bubbles = sorted(ovalContours[i:i + bubbleCount], key=x_cord_contour,reverse=False)
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
                            ovalContours = sorted(ovalContours, key=y_cord_contour, reverse=False)
                            for (q, i) in enumerate(np.arange(0, len(ovalContours), bubbleCount)):
                                bubbles = sorted(ovalContours[i:i + bubbleCount], key=x_cord_contour, reverse=False)
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
                upperimage = cv2.imread(new_folder_path+"/upper_"+filename[:-4]+".jpg")
                opened_upperimage = Image.open(new_folder_path+"/upper_"+filename[:-4]+".jpg")
                w, h = opened_upperimage.size
                frame = cv2.resize(upperimage, (w, h), interpolation=cv2.INTER_LANCZOS4)
                uppercannyFrame = getCannyFrame(frame)
                upperwarpedFrame = getWarpedFrame(uppercannyFrame, frame)
                upperadaptiveFrame = getAdaptiveThresh(upperwarpedFrame)
                upperovalContours = getOvalContours(upperadaptiveFrame)
                if (len(upperovalContours) == upperovalCount):
                    print("upper----------------")
                    print(len(upperovalContours))
                    upperovalContours = sorted(upperovalContours, key=y_cord_contour, reverse=False)
                    for (q, i) in enumerate(np.arange(0, len(upperovalContours), upperbubbleCount)):
                        upperbubbles = sorted(upperovalContours[i:i + upperbubbleCount], key=x_cord_contour,reverse=False)
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
                    if filename not in failed_images:
                        if score < 0 :
                            score = 0
                        student_score = StudentsScores.objects.update_or_create(exam=exam,student=student,score=score*question_score)
                else:
                    upperimage = cv2.imread(new_folder_path+"/upper_"+filename[:-4]+".jpg")
                    h = int(round(600 * upperimage.shape[0] / upperimage.shape[1]))
                    frame = cv2.resize(upperimage, (600, h), interpolation=cv2.INTER_LANCZOS4)
                    uppercannyFrame = getCannyFrame(frame)
                    upperwarpedFrame = getWarpedFrame(uppercannyFrame, frame)
                    upperadaptiveFrame = getAdaptiveThresh(upperwarpedFrame)
                    upperovalContours = getOvalContours(upperadaptiveFrame)
                    if (len(upperovalContours) == upperovalCount):
                        print("upper----------------")
                        print(len(upperovalContours)) 
                        upperovalContours = sorted(upperovalContours, key=y_cord_contour, reverse=False)
                        for (q, i) in enumerate(np.arange(0, len(upperovalContours), upperbubbleCount)):
                            upperbubbles = sorted(upperovalContours[i:i + upperbubbleCount], key=x_cord_contour,reverse=False)
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
                        if filename not in failed_images:
                            if score < 0 :
                                score = 0
                            student_score = StudentsScores.objects.update_or_create(exam=exam,student=student,score=score*question_score)
                    else:
                        upperimage = cv2.imread(new_folder_path+"/upper_"+filename[:-4]+".jpg")
                        h = int(round(400 * upperimage.shape[0] / upperimage.shape[1]))
                        frame = cv2.resize(upperimage, (400, h), interpolation=cv2.INTER_LANCZOS4)
                        uppercannyFrame = getCannyFrame(frame)
                        upperwarpedFrame = getWarpedFrame(uppercannyFrame, frame)
                        upperadaptiveFrame = getAdaptiveThresh(upperwarpedFrame)
                        upperovalContours = getOvalContours(upperadaptiveFrame)
                        if (len(upperovalContours) == upperovalCount):
                            upperovalContours = sorted(upperovalContours, key=y_cord_contour, reverse=False)
                            for (q, i) in enumerate(np.arange(0, len(upperovalContours), upperbubbleCount)):
                                upperbubbles = sorted(upperovalContours[i:i + upperbubbleCount], key=x_cord_contour,reverse=False)
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
                            if filename not in failed_images:
                                if score < 0 :
                                    score = 0
                                student_score = StudentsScores.objects.update_or_create(exam=exam,student=student,score=score*question_score)
                        else:
                            print('error from upper')
                            print(len(upperovalContours))
                            if filename not in failed_images:
                                failed_images.append(filename)
            
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