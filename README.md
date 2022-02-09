# ssac_final_project

**Sessac 2021 영상처리를 위한 인공지능SW개발자 양성 과정 _ 2팀 Final Project**


**Project title : 팔굽혀펴기 바른 자세 및 개수 추정** 


**team member : 이지훈(팀장), 문주현, 홍훈표**


## Directory Structure

├── fitness

│   ├── \_\_init\_\_.py

│   ├── \_\_pycache\_\_

│   ├── asgi.py

│   ├── settings.py

│   ├── urls.py

│   └── wsgi.py

├── media **\#웹캠 녹화 및 업로드시 영상이 저장되는 경로**

│   └── video

├── persistence

│   ├── db

│   └── models

├── pushup

│   ├── \_\_init\_\_.py

│   ├── \_\_pycache__

│   ├── admin.py

│   ├── apps.py

│   ├── data **\#저장된 비디오 영상을 이미지(jpg)와 포즈(json)데이터로 나누어 저장하는 장소**

│   ├── db.sqlite3

│   ├── migrations

│   ├── models.py

│   ├── psu_models **\#키포인트 추출 및 운동 추정 모델**

│   ├── templates

│   ├── tests.py

│   ├── urls.py

│   └── views.py

├── run

│   └── gunicorn.sock

└── static

│   ├──  admin

│   ├── assets

│   ├── css

│   ├── js

├── manage.py

├── db.sqlite3

├── django.crt

├── django.key


