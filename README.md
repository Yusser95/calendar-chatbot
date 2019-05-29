# calendar-chatbot
flask app to extract calendar actions (using regix) and date time (using python-duckling) from text input

# demo
https://duckling-parse-demo.herokuapp.com/

# installation
- run pip3 install requirments.txt
- set training data in examples.json
- uncomment model = uesr_model_class() in user_model.py
- uncomment model._train_models() in user_model.py
- run python3 app.py

# todo
- scrape more data for calendar intents
- train intent classification model
- extract actions info using nlp
