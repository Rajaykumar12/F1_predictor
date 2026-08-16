import logging
import sys

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)


@click.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind host.")
@click.option("--port", default=8000, show_default=True, help="Bind port.")
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload (dev mode).")
def serve(host: str, port: int, reload: bool):
    """Start the FastAPI prediction server.

    Once running, use Swagger UI at http://<host>:<port>/docs to run the data
    pipeline, train models, and get predictions — there are no other CLI
    commands.
    """
    import uvicorn

    uvicorn.run("app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    serve()
