FROM python:3.9

RUN pip install numpy


COPY . .

RUN ["python3", "main.py", "main_2.py"]