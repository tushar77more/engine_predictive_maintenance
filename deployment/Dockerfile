FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir \
    streamlit>=1.30.0 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    scikit-learn==1.4.2 \
    xgboost==1.7.6 \
    joblib==1.3.2

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
