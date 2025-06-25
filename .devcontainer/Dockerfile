# Use official Miniconda base image
FROM continuumio/miniconda3

# Set up environment
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && conda clean -a

# Set the default environment
ENV PATH /opt/conda/envs/anaconda-env/bin:$PATH

# Activate environment on shell start
RUN echo "conda activate anaconda-env" >> ~/.bashrc
