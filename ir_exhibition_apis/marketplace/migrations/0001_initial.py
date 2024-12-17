# Generated by Django 5.1.4 on 2024-12-17 13:15

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Product',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=255)),
                ('price', models.DecimalField(decimal_places=2, max_digits=10)),
                ('location', models.CharField(max_length=100)),
                ('image_url', models.URLField(blank=True, null=True)),
                ('time', models.CharField(max_length=50)),
                ('category', models.CharField(choices=[('Cars', 'Cars'), ('Mobiles', 'Mobiles')], max_length=10)),
            ],
        ),
    ]
