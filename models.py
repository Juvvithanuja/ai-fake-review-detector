from django.db import models
from django.contrib.auth.models import User

class ReviewRecord(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    review_text = models.TextField()
    prediction = models.CharField(max_length=50)
    real_confidence = models.FloatField()
    fake_confidence = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user} - {self.prediction}"
