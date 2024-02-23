from django.db import models
# Create your models here.

import os

def exam_file_path(instance, filename):
    """
    Returns the path where the file should be uploaded for the given Files instance.
    The file will be uploaded to a directory named after the exam's ID or title.
    """
    exam = instance.exam
    exam_id = str(exam.id)
    exam_title = exam.title.replace(' ', '_')
    # return os.path.join('examfiles', exam_id + '_' + exam_title, filename)
    return os.path.join('examfiles', exam_id , filename)

class Exam(models.Model):
    title = models.CharField(max_length=200, verbose_name='عنوان الامتحان') 
    examiner = models.CharField(max_length=200, verbose_name=' جهة الامتحان', null=True, blank=True) 
    questions_number = models.PositiveIntegerField(verbose_name='عدد الاسئلة')
    answeres_number = models.PositiveSmallIntegerField(verbose_name='عدد الاجابات فى السؤال الواحد', default=5)
    question_score = models.PositiveSmallIntegerField(verbose_name='درجة السؤال')
    created_at = models.DateTimeField(auto_now=True, auto_now_add=False, verbose_name='تاريخ الانشاء')

    def totalscore(self):
        total_score = self.questions_number * self.question_score
        return total_score
    def examanalytics(self):
        students_scores = self.studentsscores_set.all()
        scores = []
        success = []
        failed = []
        scores_AVG=0
        successfull_number=0
        failed_number=0
        gratest_score=0
        minimum_score=0
        success_percentage=0
        try:
            for score in students_scores.values('score'):
                scores.append(score['score'])
                if score['score'] >= self.totalscore()/2:
                    success.append(score['score'])
                else:
                    failed.append(score['score'])
            scores_AVG = sum(scores)/len(scores)
            
            successfull_number = len(success)
            failed_number = len(failed)
            scores.sort()
            # best_3_scores = students_scores[0:3]
            gratest_score = scores[-1]
            minimum_score = scores[0]
            success_percentage = (len(success)/len(scores))*100
        except:
            print("nothing")
        return students_scores,successfull_number,failed_number,scores_AVG,gratest_score,minimum_score,success_percentage
    def __str__(self):
        return self.title
    class Meta:
        verbose_name_plural = 'الامتحانات'
        ordering = ['created_at']

class Files(models.Model):
    exam = models.ForeignKey(Exam, on_delete=models.CASCADE,null=True)
    file = models.FileField(upload_to=exam_file_path)

class Students(models.Model):
    student_name = models.CharField(max_length=200,verbose_name='اسم الطالب') 
    military_number = models.CharField(max_length=200,verbose_name='الرقم العسكرى') 

    def __str__(self):
        return self.student_name+"  "+self.military_number
    class Meta:
        verbose_name_plural = 'الطلاب'
        ordering = ['student_name']

class StudentsScores(models.Model):
    student = models.ForeignKey(Students,on_delete=models.CASCADE,verbose_name=' الطالب')
    exam = models.ForeignKey(Exam,on_delete=models.CASCADE,verbose_name=' الامتحان')
    score = models.PositiveIntegerField(verbose_name='الدرجة')
    # succase = models.CharField(max_length=10,null=True,blank=True)
    # def succasefunc(self):
    #     if self.score >= self.exam.totalscore :
    #         self.succase = 'succeeded'
    #         return 'succeeded' 
    #     else : 
    #         self.succase = 'failed'
    #         return 'failed'
    def __str__(self):
        return self.exam.title +"  "+self.student.student_name 
    class Meta:
        verbose_name_plural = 'درجات الطلاب'
        ordering = ['-score']
































