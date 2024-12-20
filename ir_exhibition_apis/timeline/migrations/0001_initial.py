# Generated by Django 5.1.4 on 2024-12-17 13:15

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='HistoricalPeriod',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('start_year', models.IntegerField()),
                ('end_year', models.IntegerField()),
                ('description', models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name='HistoricalContent',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('content', models.TextField()),
                ('period', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='timeline.historicalperiod')),
            ],
        ),
    ]
