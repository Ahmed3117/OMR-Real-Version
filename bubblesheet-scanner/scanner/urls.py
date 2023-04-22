from django.urls import path
from .views import printscore,getCurrentExamDegrees,home,ExamsFilter
app_name="scanner"
urlpatterns = [
    path('',home,name="home"),
    path('printscore',printscore,name="printscore"),
    path('examscores/<int:pk>/',getCurrentExamDegrees,name="examscores"),
    path('specificexamscores',ExamsFilter,name="specificexamscores"),
    
]




    