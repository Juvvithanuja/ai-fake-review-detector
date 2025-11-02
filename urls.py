from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
    path('login/', views.user_login, name='login'),
    path("register/", views.register, name="register"),
    path('history/', views.history, name='history'),
    path('logout/', views.user_logout, name='logout'),
    path('users/', views.show_users, name='show_users'),

]
