FROM zironycho/pytorch:1120-cpu-py38

WORKDIR /workspace

#RUN pip3 install torchvision==0.14.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu && rm -rf /root/.cache/pip

COPY gold/gradio/requirements.txt .

RUN pip3 install -r requirements.txt && rm -rf /root/.cache/pip

#COPY configs /workspace/configs/
#COPY ckpt /workspace/ckpt/
#COPY gold /workspace/gold/

COPY . .

RUN pip3 install -e .

EXPOSE 8080

CMD [ "python3", "gold/gradio/demo_vit_jit.py"]