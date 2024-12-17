from django.db import models

class Product(models.Model):
    CATEGORY_CHOICES = [
        ('Cars', 'Cars'),
        ('Mobiles', 'Mobiles')
    ]
    
    title = models.CharField(max_length=255)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    location = models.CharField(max_length=100)
    image_url = models.URLField(null=True, blank=True)
    time = models.CharField(max_length=50)
    category = models.CharField(max_length=10, choices=CATEGORY_CHOICES)

    def __str__(self):
        return self.title