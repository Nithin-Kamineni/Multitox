from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser
from django.http.response import JsonResponse
import sqlite3
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors
import numpy as np
import pandas as pd
from keras.models import load_model

from django.core.files.storage import default_storage
# from aipbpk.Handlers.MLModelLoaders import getModels
# from aipbpk.Handlers.DataLoader import GetData
from aipbpk.Handlers.MLModel import mLModel


import os
from dotenv import load_dotenv
load_dotenv()

# Create your views here.
@csrf_exempt
def test(request,id=0):
    if request.method=='GET':
        print(os.environ.get('DB_HOST'))
        print(os.environ.get('tesval'))
        return JsonResponse("Get request",safe=False)
    elif request.method=='POST':
        return JsonResponse("Failed to Add",safe=False)
    elif request.method=='PUT':
        return JsonResponse("Updated Successfully",safe=False)
        return JsonResponse("Failed to Update")
    elif request.method=='DELETE':
        return JsonResponse("Deleted Successfully",safe=False)
        
@csrf_exempt
def getPrediction(request, id=0):
    if request.method == 'POST':
        request_body = JSONParser().parse(request)
        toxdata = request_body.get("toxdata")

        caslist = request_body.get("caslist").rstrip(', ').split(', ')

        print(toxdata, caslist)

        if(toxdata=="include"):
            y_hatNeuroInpre = mLModel.GetNeuroInluded()
            y_hatRemInpre = mLModel.GetRemInluded()

            index = mLModel.GetIndexes(SMILES=caslist[0])
            # if(index==False):

            print(len(y_hatNeuroInpre), index)

            y_hatNeurosIn = y_hatNeuroInpre[index[0]:index[0]+1]
            y_hatRemsIn = y_hatRemInpre[index[0]:index[0]+1]

            print(len(y_hatRemsIn), index)
            print('---------------------------------')
            print(y_hatRemsIn)

            y_hatRemsIn[0]['Neurotoxicity'] = y_hatNeurosIn[0]['Neurotoxicity']
            return JsonResponse({'y_hatRem':y_hatRemsIn }, safe=False)
        else:
            y_hatRems = []
            for smiles in caslist:
                # print('cas-ex',cas)
                y_hatNeuro = mLModel.GetNeuroNotInluded([smiles])
                y_hatRem = mLModel.GetRemNotInluded([smiles])

                if(y_hatRem==None):
                    return JsonResponse({'y_hatRem':False, }, safe=False)

                y_hatRem[0]['Neurotoxicity'] = y_hatNeuro[0]['Neurotoxicity']

                y_hatRems.append(y_hatRem[0])
            return JsonResponse({'y_hatRem':y_hatRems, }, safe=False)
        

