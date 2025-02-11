from django.urls import path
from aipbpk import views

from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('test', views.test),
    path('prediction', views.getPrediction),
    
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
