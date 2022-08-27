#!/bin/bash
sudo docker build -t serv_imgcap .
sudo docker rm -f serv_imgcap
sudo docker run -it -p 55558:80 -e GUNICORN_CONF="/app/gunicorn_conf.py" --name=serv_imgcap serv_imgcap
