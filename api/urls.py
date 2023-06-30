from . import views
from django.urls import path
urlpatterns = [
    path('', views.classify_image, name='classify')
]