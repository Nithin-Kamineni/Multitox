import requests
import json
import os
from dotenv import load_dotenv
load_dotenv()


def send_post_request(predKmax, BW, Dose):
    url = os.environ.get('Rserver')  # Replace with the actual URL
    data = {"predKmax": list(predKmax),"BW":BW,"Dose":Dose}  # Replace with your data
    # data = json.dumps({"predKmax": predKmax})
    url = url + "predkmaxcsv"
    # print("............came here...........................23232356")
    # print(data)
    # print(predKmax[0])
    # print(type(predKmax[0]['KTRES50']))
    # print("datarequest================================")
    # url='http://rserver_container:9000/patrol?messg=nithin'
    print("11111111111111111111111111111111111111115672")
    print(url)
    print(data)
    print("11111111111111111111111111111111111111115672")
    response = requests.post(url, verify=False, json=data)  # Sending JSON data in the request
    # response = requests.get(url)
    # return {"list":'hell'}
    # Check the response
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    if response.status_code == 200:  # Successful response
        response_data = response.json()
        # print("Response:", response_data)
        # response_data["list"] = response_data["messg"]
        return response_data
    else:
        print("Request failed with status code:", response.status_code)

# Call the function to send the POST request

