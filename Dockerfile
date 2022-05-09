FROM python:3.9-slim
WORKDIR /tmp
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN rm requirements.txt
WORKDIR /project
