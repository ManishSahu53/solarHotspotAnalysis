FROM python:3.7.8-slim

# remember to expose the port your app'll be exposed on.
EXPOSE 8080
RUN pip install -U pip

RUN apt update && apt install -y git

COPY requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt

# RUN git clone https://github.com/francesconazzaro/streamlit.git
# RUN pip install streamlit

# copy into a directory of its own (so it isn't in the toplevel dir)
COPY . /app
WORKDIR /app

# run it!
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0", "--logger.level=error"]