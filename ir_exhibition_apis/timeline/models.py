from django.db import models

class HistoricalPeriod(models.Model):
    name = models.CharField(max_length=100)
    start_year = models.IntegerField()
    end_year = models.IntegerField()
    description = models.TextField()

    def __str__(self):
        return self.name

class HistoricalContent(models.Model):
    period = models.ForeignKey(HistoricalPeriod, on_delete=models.CASCADE)
    content = models.TextField()

    def __str__(self):
        return f"{self.period.name} Content"