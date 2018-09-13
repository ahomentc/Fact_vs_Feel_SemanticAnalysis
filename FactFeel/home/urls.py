from django.conf.urls import url
from . import views

app_name = 'home'

urlpatterns = [
    url(r'^$', views.IndexView, name='index'),
    
    # send the text
    url(r'^make_prediction',views.make_prediction,name='make_prediction'),
]
