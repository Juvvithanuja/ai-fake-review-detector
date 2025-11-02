'''from django.contrib import admin
from .models import ReviewRecord

admin.site.register(ReviewRecord)'''
from django.contrib import admin
from .models import ReviewRecord

@admin.register(ReviewRecord)
class ReviewRecordAdmin(admin.ModelAdmin):
    list_display = ('user', 'review_text', 'prediction', 'real_confidence', 'fake_confidence', 'created_at')
    search_fields = ('user__username', 'review_text')
    list_filter = ('prediction', 'created_at')
