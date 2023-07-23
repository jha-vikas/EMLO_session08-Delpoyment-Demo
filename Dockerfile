FROM zironycho/pytorch:1120-cpu-py38

WORKDIR /workspace

COPY gold/gradio/requirements.txt .

RUN pip3 install -r requirements.txt && rm -rf /root/.cache/pip


COPY . .

RUN pip3 install -e .

EXPOSE 8080

CMD [ "python3", "gold/gradio/demo_vit_jit.py"]