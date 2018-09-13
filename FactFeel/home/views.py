from django.shortcuts import render
from django.http import HttpResponse
import sys
sys.path.insert(0, '/Users/andrei/fact_vs_feel')
import makePrediction
import json

def IndexView(request):
    return render(request, 'home/index.html',)

def make_prediction(request):

    # !! in the future make it so that either the position holders or community vote
    # !! or anybody who has the position or is an admin can grant access

    text_data = request.POST['text_data']
    model = makePrediction.FactOrFeelModel()
    percentages = model.evaluateText(text_data)

    return HttpResponse(json.dumps(percentages))