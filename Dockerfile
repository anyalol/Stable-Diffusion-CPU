FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
COPY . /app
WORKDIR /app
RUN apt-get update
RUN apt-get install -y git
RUN pip install -r requirements.txt
ENV GRADIO_SERVER_PORT=8080
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV CUDA_VISIBLE_DEVICES=""
EXPOSE 8080
RUN python3 procure.py
ENTRYPOINT ["python3", "app.py"]
