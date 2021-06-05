# lightweight python
FROM python:3.7-slim

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
    apt-get install --no-install-recommends --assume-yes \
      protobuf-compiler

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN ls -la $APP_HOME/

# Install dependencies
RUN pip install -r requirements.txt
WORKDIR models/research
RUN protoc object_detection/protos/*.proto --python_out=.
RUN cp object_detection/packages/tf1/setup.py .
RUN python -m pip install .
WORKDIR ../..

# Run the streamlit on container startup
CMD [ "streamlit", "run","waldo_detector.py" ]