FROM python:3.11.11

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

# Expose port 8000
EXPOSE 8000

CMD ["uvicorn", "aip:app", "--host", "0.0.0.0", "--port", "8000"]
