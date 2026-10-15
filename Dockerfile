FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /code

COPY pyproject.toml uv.lock README.md /code/
COPY ./src /code/src

RUN uv sync --frozen

RUN uv run python -m spacy download en_core_web_lg

COPY ./app /code/app
COPY ./helper_lib /code/helper_lib
COPY ./models /code/models
COPY main.py /code/

CMD ["uv", "run", "fastapi", "run", "main.py", "--port", "80"]