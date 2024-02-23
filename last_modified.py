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
uppertotalquestionCount = 6
upperbubbleCount = 10
upperovalCount = uppertotalquestionCount * upperbubbleCount
BubbleSigned = 0
score = 0
secretenumber = ''
dimentions = [0,400,300,280,310,250,290,200,600,220,320]
upperdimentions=[0,600,400,500,420,300,320]
threshing_numbers=[190,198,150,127,100,200]
failed_images = []

def getCannyFrame( frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
    frame = cv2.Canny(gray, 127, 255)
    # frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 1)
    return frame


def getAdaptiveThresh( frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    adaptiveFrame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 1)
    # adaptiveFrame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 1)
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
        if (len(approx) > 10 and w / h <= 1.4 and w / h >= 0.8):

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


def split_image_to_upper_lower(uploaded_files_path,new_folder_path,filename,upper_cut_hight):
    input_image = Image.open(uploaded_files_path+"/"+filename)
    input_image = input_image.resize((794, 1123), resample=Image.BICUBIC)
    # Get the dimensions of the input image
    width, height = input_image.size
    # Calculate the coordinates to split the image in half
    split_coord = height // upper_cut_hight
    # Split the image into upper and lower halves
    upper_half = input_image.crop((0, 0, width, split_coord))
    lower_half = input_image.crop((0, split_coord, width, height))
    # Convert the images to RGB mode
    upper_half = upper_half.convert('RGB')
    lower_half = lower_half.convert('RGB')
    # Save the two halves as separate files
    upper_half.save(new_folder_path+"/upper_"+filename[:-4]+".jpg")
    lower_half.save(new_folder_path+"/lower_"+filename[:-4]+".jpg")
        
def split_lowerimage_to_left_right(new_folder_path,filename,cut_distance):
    lower_input_image = Image.open(new_folder_path+"/lower_"+filename[:-4]+".jpg")
    # Get the dimensions of the input image
    width, height = lower_input_image.size
    # Calculate the coordinates to split the image in half
    split_coord = width // cut_distance
    # Split the image into upper and lower halves
    left_half = lower_input_image.crop((0, 0, split_coord, height))
    right_half = lower_input_image.crop((split_coord, 0, width, height))
    # Save the two halves as separate files
    left_half.save(new_folder_path+"/left_"+filename[:-4]+".jpg") 
    right_half.save(new_folder_path+"/right_"+filename[:-4]+".jpg")
        
def thresholding_image(uploaded_files_path,filename,threshold_degree):
    '''  
    apply threeshold (convert the image to white and black) 
    if the pixel color is < threshold_degree (190 for example) , it is cnverted to black (0) 
    , else it is converted to white (255)
    
    ''' 
    img = Image.open(uploaded_files_path+"/"+filename)
    threshold = threshold_degree
    img = img.point(lambda x: 0 if x < threshold else 255)
    img = img.convert("1")
    img.save(uploaded_files_path+"/"+filename)

    # img = Image.open(uploaded_files_path+"/"+filename)
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 1)
    # img.save(uploaded_files_path+"/"+filename)

def thresholding_image_part(image_part_path,threshold_degree):

    img = Image.open(image_part_path)
    threshold = threshold_degree
    # threshold = 150
    if threshold != 0 :
        img = img.point(lambda x: 0 if x < threshold else 255)
        img = img.convert("1")
    img.save(image_part_path[:-4]+str(threshold_degree)+'.jpg')


# def get_ovalcontours_from_image_with_normal_dimntions(image_path):
#     imagee = cv2.imread(image_path)
#     h, w, _ = imagee.shape
#     frame = cv2.resize(imagee, (w, h), interpolation=cv2.INTER_LANCZOS4)
#     cannyFrame = getCannyFrame(frame)
#     warpedFrame = getWarpedFrame(cannyFrame, frame)
#     adaptiveFrame = getAdaptiveThresh(warpedFrame)
#     ovalContours = getOvalContours(adaptiveFrame)
#     return ovalContours,adaptiveFrame

def get_ovalcontours_from_image_with_list_of_dimentions(image_path,dim):
    imagee = cv2.imread(image_path)
    h=w=0
    frame = ''
    if dim == 0 :
        h, w, _ = imagee.shape
        frame = cv2.resize(imagee, (w, h), interpolation=cv2.INTER_LANCZOS4)
    else:
        h = int(round(dim * imagee.shape[0] / imagee.shape[1]))
        frame = cv2.resize(imagee, (dim, h), interpolation=cv2.INTER_LANCZOS4)
    cannyFrame = getCannyFrame(frame)
    warpedFrame = getWarpedFrame(cannyFrame, frame)
    adaptiveFrame = getAdaptiveThresh(warpedFrame)
    ovalContours = getOvalContours(adaptiveFrame)
    return ovalContours,adaptiveFrame

def score_image_part(ovalContours,adaptiveFrame,bubbleCount,ANSWER_KEY):
    global score
    global BubbleSigned
    ovalContours = sorted(ovalContours, key=y_cord_contour, reverse=False)
    for (q, i) in enumerate(np.arange(0, len(ovalContours), bubbleCount)):

        bubbles = sorted(ovalContours[i:i + bubbleCount], key=x_cord_contour,reverse=False)
        
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
                BubbleSigned += 1
                if (answer == j):
                    # And calculate score
                    score += 1

def read_military_number(ovalContours,adaptiveFrame,upperbubbleCount):
    global secretenumber
    ovalContours = sorted(ovalContours, key=y_cord_contour, reverse=False)
    for (q, i) in enumerate(np.arange(0, len(ovalContours), upperbubbleCount)):
        upperbubbles = sorted(ovalContours[i:i + upperbubbleCount], key=x_cord_contour,reverse=False)
        for (j, c) in enumerate(upperbubbles):
            area = cv2.contourArea(c)
            uppermask = np.zeros(adaptiveFrame.shape, dtype="uint8")
            cv2.drawContours(uppermask, [c], -1, 255, -1)
            uppermask = cv2.bitwise_and(adaptiveFrame, adaptiveFrame, mask=uppermask)
            total = cv2.countNonZero(uppermask)
            x, y, w, h = cv2.boundingRect(c)
            upperisBubbleSigned = ((float)(total) / (float)(area)) > 1
            if (upperisBubbleSigned):
                secretenumber += str(j)



def printscore(request):
    
    
    global failed_images
    failed_images = []
    global score
    score = 0
    global BubbleSigned
    BubbleSigned = 0
    global secretenumber
    secretenumber = ''
    global dimentions
    global upperdimentions
    global upperovalCount


    # tf_questions_number = 10
    # tf_oval_number = tf_questions_number *2 # 10*2 = 20
    # choices_questions_number = 40
    # toal_num_of_questions = tf_questions_number + choices_questions_number
    # choices_oval_number = choices_questions_number *4 # 40*4 = 160


    exam_title = request.POST.get("examtitle")
    examiner = request.POST.get("examiner")
    c_questionCount = int(request.POST.get("questionsnumber"))
    tf_questions_number = request.POST.get("tfquestionsnumber") or 0
    tf_questions_number = int(tf_questions_number)
    totalquestionCount = c_questionCount + tf_questions_number
    tf_oval_number = tf_questions_number *2 # 10*2 = 20
    bubbleCount = int(request.POST.get("answeres_number"))
    ovalCount = totalquestionCount * bubbleCount
    question_score = int(request.POST.get("onequestionscore"))
    files = request.FILES.getlist('files')
    exam = Exam.objects.create(title=exam_title, examiner=examiner, questions_number=totalquestionCount, answeres_number=bubbleCount, question_score=question_score)
    exam.save()
    for f in files:
        file_obj = Files.objects.create(file=f,exam=exam)
        file_obj.save()
    uploaded_files_path = os.path.dirname(file_obj.file.path)

    ANSWER_KEY1 = {}
    ANSWER_KEY2 = {}
    ANSWER_KEY3 = {}
    i=0
    
    
    for box in range(0,int(totalquestionCount/2)):
        ans=request.POST.get("box-"+str(box)+"-radio")
        if ans == None :
            ANSWER_KEY1[box] = ans
        else:
            ANSWER_KEY1[box] = int(ans)
    print(ANSWER_KEY1)
    for box in range(int(totalquestionCount/2),totalquestionCount):
        ans=request.POST.get("box-"+str(box)+"-radio")
        if ans == None :
            ANSWER_KEY2[box-int(totalquestionCount/2)] = ans
        else:
            ANSWER_KEY2[box-int(totalquestionCount/2)] = int(ans)
    print(ANSWER_KEY2)
    for box in range(int(totalquestionCount/2) - tf_questions_number,int(totalquestionCount/2)):
        ANSWER_KEY3[i] = ANSWER_KEY2[box]
        i+=1
    print(ANSWER_KEY3)


    folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  
    new_folder_path = os.path.join(uploaded_files_path, folder_name)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    
    for filename in os.listdir(uploaded_files_path): 
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".PNG") or filename.endswith(".pdf") or filename.endswith(".PDF") :
            try:
                print(filename)
                # thresholding_image(uploaded_files_path,filename,198)
                split_image_to_upper_lower(uploaded_files_path,new_folder_path,filename,3.8)
                split_lowerimage_to_left_right(new_folder_path,filename,2)
                #-------------------------------------------------------------------------------------------------

                # left_image
                print("left---------------------" )
                left_image_path = new_folder_path+"/left_"+filename[:-4]+".jpg" 
                case = 0 # 0 means that ovalContours not be found yet.
                for threshing_number in threshing_numbers:
                    try:
                        if case == 1 : # 1 means that ovalContours have been found.
                            break
                        thresholding_image_part(left_image_path,threshing_number)
                        image_part_path = new_folder_path+"/left_"+filename[:-4]+str(threshing_number)+".jpg" 
                        for dim in dimentions:
                            try:
                                ovalcontours_from_image_with_list_of_dimentions=get_ovalcontours_from_image_with_list_of_dimentions(image_part_path,dim)
                                ovalContours = ovalcontours_from_image_with_list_of_dimentions[0]
                                adaptiveFrame = ovalcontours_from_image_with_list_of_dimentions[1]
                                
                                print('threshing_num =>'+str(threshing_number)+' => dim => '+ str(dim) +'=> ' + str(len(ovalContours)))
                                if (len(ovalContours) == ovalCount/2):
                                    score_image_part(ovalContours,adaptiveFrame,bubbleCount,ANSWER_KEY1)
                                    case = 1
                                    break
                                else:
                                    if (dimentions.index(dim) == len(dimentions)-1) and (threshing_numbers.index(threshing_number) == len(threshing_numbers)-1):
                                        print('error from left') 
                                        if filename not in failed_images:
                                            failed_images.append(filename)
                            except:
                                print('out from =>' + str(dim))                
                    except:
                        print('not good =>' + str(threshing_number))  

                # right_image
                print("right----------------" )
                right_image_path = new_folder_path+"/right_"+filename[:-4]+".jpg" 
                image_part_path = new_folder_path+"/right_"+filename[:-4]+str(150)+".jpg" 
                thresholding_image_part(right_image_path,150)
                for dim in dimentions:
                    ovalcontours_from_image_with_list_of_dimentions=get_ovalcontours_from_image_with_list_of_dimentions(image_part_path,dim)
                    ovalContours = ovalcontours_from_image_with_list_of_dimentions[0]
                    adaptiveFrame = ovalcontours_from_image_with_list_of_dimentions[1]                        
                    print('dim => '+ str(dim) +'=> ' + str(len(ovalContours)))
                    if (len(ovalContours) == (int(ovalCount/2) - tf_oval_number)):
                        print(len(ovalContours))
                        c_ovalContours = ovalContours[tf_oval_number:int(ovalCount/2) - tf_oval_number]
                        tf_ovalContours = ovalContours[0:tf_oval_number]
                        score_image_part(c_ovalContours,adaptiveFrame,bubbleCount,ANSWER_KEY2)
                        if len(ANSWER_KEY3) > 0 :                    
                            score_image_part(tf_ovalContours,adaptiveFrame,2,ANSWER_KEY3)
                        break
                    else:
                        if dimentions.index(dim) == len(dimentions)-1 :
                            print('error from right')
                            print(len(ovalContours))
                            if filename not in failed_images:
                                failed_images.append(filename)

                # if BubbleSigned > 50 :
                #     score = score - (BubbleSigned-50)
                
                # upper_image
                upper_image_path = new_folder_path+"/upper_"+filename[:-4]+".jpg" 
                image_part_path = new_folder_path+"/upper_"+filename[:-4]+str(0)+".jpg" 
                threshing_number = 0
                thresholding_image_part(upper_image_path,threshing_number)
                for dim in upperdimentions:
                    ovalcontours_from_image_with_list_of_dimentions=get_ovalcontours_from_image_with_list_of_dimentions(upper_image_path,dim)
                    if threshing_number != 0 :
                        ovalcontours_from_image_with_list_of_dimentions=get_ovalcontours_from_image_with_list_of_dimentions(image_part_path,dim)
                    
                    # ovalcontours_from_image_with_list_of_dimentions = 0
                    # if threshing_number != 0 :
                    #     ovalcontours_from_image_with_list_of_dimentions=get_ovalcontours_from_image_with_list_of_dimentions(upper_image_path,dim)
                    
                    # else:
                    #     ovalcontours_from_image_with_list_of_dimentions=get_ovalcontours_from_image_with_list_of_dimentions(image_part_path,dim)
                    ovalContours = ovalcontours_from_image_with_list_of_dimentions[0]
                    adaptiveFrame = ovalcontours_from_image_with_list_of_dimentions[1]
                    print('dim => '+ str(dim) +'=> ' + str(len(ovalContours)))
                    if (len(ovalContours) == upperovalCount):
                        print("upper----------------" )
                        print(len(ovalContours))
                        read_military_number(ovalContours,adaptiveFrame,upperbubbleCount)
                        try:
                            student = Students.objects.get(military_number=secretenumber)
                        except:
                            student = Students.objects.create(military_number=secretenumber,student_name = secretenumber)
                        if filename not in failed_images:
                            if score < 0 :
                                score = 0
                            student_score = StudentsScores.objects.update_or_create(exam=exam,student=student,score=score*question_score)
                        break
                    else:
                        if upperdimentions.index(dim) == len(upperdimentions)-1 :
                            print('error from upper')
                            print(len(ovalContours))
                            if filename not in failed_images:
                                failed_images.append(filename)
                    
            except:
                if filename not in failed_images:
                    print("outer errorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
                    failed_images.append(filename)

        score=0
        BubbleSigned = 0
        secretenumber = ''
    return redirect('scanner:examscores' ,exam.id)


def getCurrentExamDegrees(request,pk):
    global failed_images
    exam =''
    students_scores = []
    if pk == 0 :
        StudentsScores
        students_scores = StudentsScores.objects.all()
    else:
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
