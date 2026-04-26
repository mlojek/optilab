FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /optilab

COPY . .

RUN uv sync --frozen

ENTRYPOINT ["/bin/bash"]
