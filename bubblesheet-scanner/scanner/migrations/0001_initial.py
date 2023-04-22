# Generated by Django 4.2 on 2023-04-22 22:08

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Exam',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=200, verbose_name='عنوان الامتحان')),
                ('examiner', models.CharField(blank=True, max_length=200, null=True, verbose_name=' الممتحن')),
                ('questions_number', models.PositiveIntegerField(verbose_name='عدد الاسئلة')),
                ('answeres_number', models.PositiveSmallIntegerField(default=5, verbose_name='عدد الاجابات فى السؤال الواحد')),
                ('question_score', models.PositiveSmallIntegerField(verbose_name='درجة السؤال')),
                ('created_at', models.DateTimeField(auto_now=True, verbose_name='تاريخ الانشاء')),
            ],
            options={
                'verbose_name_plural': 'الامتحانات',
                'ordering': ['created_at'],
            },
        ),
        migrations.CreateModel(
            name='Files',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file', models.FileField(upload_to='2023-04-15-08-12')),
            ],
        ),
        migrations.CreateModel(
            name='Students',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('student_name', models.CharField(max_length=200, verbose_name='اسم الطالب')),
                ('military_number', models.CharField(max_length=200, verbose_name='الرقم العسكرى')),
            ],
            options={
                'verbose_name_plural': 'الطلاب',
                'ordering': ['student_name'],
            },
        ),
        migrations.CreateModel(
            name='StudentsScores',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('score', models.PositiveIntegerField(verbose_name='الدرجة')),
                ('exam', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='scanner.exam', verbose_name=' الامتحان')),
                ('student', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='scanner.students', verbose_name=' الطالب')),
            ],
            options={
                'verbose_name_plural': 'درجات الطلاب',
                'ordering': ['-score'],
            },
        ),
        migrations.AddField(
            model_name='exam',
            name='files',
            field=models.ManyToManyField(related_name='exams', to='scanner.files', verbose_name='اضف صور للتصحيح'),
        ),
    ]