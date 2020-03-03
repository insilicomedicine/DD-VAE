FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN mkdir -p /code

RUN set -ex \
    && apt-get update \
    && apt-get install -y git vim less wget \
        tmux libxrender1 libxext6

RUN set -ex \
    && wget https://repo.continuum.io/miniconda/Miniconda3-4.7.10-Linux-x86_64.sh \
    && /bin/bash Miniconda3-4.7.10-Linux-x86_64.sh -f -b -p /opt/miniconda

ENV PATH /opt/miniconda/bin:$PATH

RUN conda install -y numpy=1.17.2 \
                     scipy=1.3.1 \
                     scikit-learn=0.20.3 \
                     matplotlib=3.1.1 \
                     pandas=0.25.1 \
                     notebook=6.0.0 \
                     networkx=2.3 \
                     ipywidgets=7.5.1

RUN conda install -y -c pytorch cudatoolkit=9.0 pytorch=1.1.0 torchvision=0.2.1

RUN conda install -y -c rdkit rdkit=2019.03.4

RUN pip install Theano==1.0.4 molsets==0.2 tensorboardX==1.9 cairosvg==2.4.2 tqdm==4.42.0

ADD . /code/dd_vae

RUN cd /code/dd_vae && python setup.py install

CMD [ "/bin/bash" ]
