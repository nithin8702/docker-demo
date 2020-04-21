FROM continuumio/anaconda3
COPY ./flask_demo /usr/local/python/
EXPOSE 6005
WORKDIR /usr/local/python/
RUN pip install -r requirements.txt
CMD python flask_predict_api.py