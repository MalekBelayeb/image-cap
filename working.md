- Before the server is started there is a prestart.sh script which will run warmup.py code
- This basically sends a request containing 1x1 pixel image to facilitate one time model download before the actual client requests hit.