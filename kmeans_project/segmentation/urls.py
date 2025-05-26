from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('recolor/', views.recolor, name='recolor'),
    path('undo/', views.undo, name='undo'),
    path('download/', views.download_image, name='download_image'),
]