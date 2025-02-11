# creating venv
python -m venv pbpkv1

# to get into virtual env
.\pbpkv1\Scripts\activate
python DjangoAPI/manage.py runserver 0.0.0.0:8022

pip install -r requirements.txt



pip freeze > requirements.txt