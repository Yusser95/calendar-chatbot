# coding: utf-8
from flask import Flask,session, request, flash, url_for, redirect, render_template, abort ,g, send_from_directory

import os
import json
from flask import Markup
from operator import itemgetter
from flask import make_response
import time
from datetime import datetime, timedelta
import re
from urllib.parse import urlparse
import requests
import sys
from flask import  jsonify
from duckling import DucklingWrapper, Dim

from flask_apscheduler import APScheduler

######.  init app
app = Flask(__name__)
app.secret_key = 'y#S%bbdEErdsbjk'
app.debug = True
cwd = os.getcwd()


scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()



def ping_server():
	print("ping erver")
	r = requests.get("https://duckling-parse-demo.herokuapp.com/")
	print(r)


scheduler.add_job(func=ping_server, trigger='interval',minutes=int(5),id="ping_server")


duckling_wrapper = DucklingWrapper(parse_datetime=True)


def duckling_parse(text):    
    result = duckling_wrapper.parse_time(text)
    return result
    
def intent_ner_parse(text):
    remind = ['remind','reminder']
    alarm = ["Wake","alarm"]
    
    
    entities = {"remind":remind,"alarm":alarm}

    ner_pattern = "|".join([str("(?P<"+k+">"+str("|".join(entities[k]))+")") for k in entities])


    ner_r = re.compile(pattern=ner_pattern, flags=re.IGNORECASE)

    temp = [{m.lastgroup : m.group()} for m in ner_r.finditer(text)]
    if temp:
        return list(temp[0].keys())[0]
    return "unknown"





@app.route("/user/duckling/parse" , methods =['GET',"POST"])
def instance_name_validator():
	response = {'data':'null'}
	text = request.args.get('text')
	print(text)


	response['data'] = duckling_parse(text)
	response['intent'] = intent_ner_parse(text)


	return jsonify(response)





@app.route("/" , methods =["GET"])
def main():
	return render_template('user/index.html')




if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True, threaded=True)
