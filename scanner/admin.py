from django.contrib import admin
from .models import Exam,Students,StudentsScores
from django.contrib.auth.models import Group
# Register your models here.
admin.site.register(Exam)
admin.site.register(Students)
admin.site.register(StudentsScores)

admin.site.unregister(Group)

admin.site.site_header = "منظومة الامتحانات"
admin.site.site_title = "منظومة الامتحانات"









