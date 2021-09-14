FROM python:3.9

RUN pip install numpy


COPY . .

RUN ["python3", "main_2D.py", "main_3D.py"]