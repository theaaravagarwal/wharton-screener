"""Standalone public interface; delegate all screening to the shared engine."""

import json
import platform
from pathlib import Path
from typing import Annotated

import psutil
import typer

from wharton import screen_service, standalone

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command()
def doctor():
    """Inspect local resources without credentials or provider requests."""
    typer.echo(json.dumps({"python": platform.python_version(), "platform": platform.system(),
                           "cpus": psutil.cpu_count(),
                           "available_memory_gib": psutil.virtual_memory().available / 2**30}))


@app.command("self-test")
def self_test():
    typer.echo(json.dumps(standalone.self_test()))


@app.command("init-screen")
def initialize(root: Path = Path("runs/standalone"),
               exchange: Annotated[list[str], typer.Option("--exchange")] = ["US"],
               env_file: Path = Path(".env"), live: bool = False, max_credits: int = 500):
    typer.echo(json.dumps(standalone.initialize(root, exchange, env_file, live=live,
                                              max_credits=max_credits)))


@app.command("screen")
def screen(config: Annotated[Path, typer.Option("--config")], as_of: str | None = None):
    typer.echo(screen_service.run(config, as_of=as_of))


@app.command("screen-status")
def status(root: Path):
    typer.echo(json.dumps(screen_service.status(root)))


@app.command("verify-screen")
def verify(path: Path):
    typer.echo(json.dumps(screen_service.verify(path)))


@app.command("verify-install")
def verify_install_command(
    output: Path = Path("runs/verification/install.json"),
    require_gpu: bool = False,
    device: str = "auto",
    live: bool = False,
    root: Path = Path("runs/standalone"),
    env_file: Path = Path(".env"),
):
    """Verify requested hardware/API paths; failed checks exit nonzero with a receipt."""
    from wharton.runtime_verify import verify_install

    try:
        typer.echo(json.dumps(verify_install(output, require_gpu=require_gpu, device=device,
                                            live=live, root=root, env_file=env_file)))
    except (ValueError, RuntimeError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from None


@app.command("package-results")
def package_results_command(run: Path, output: Path):
    """Create the private results archive to send back to the research owner."""
    from wharton.result_bundle import package_results

    typer.echo(json.dumps({"archive": str(package_results(run, output))}))


@app.command("verify-results")
def verify_results_command(path: Path):
    from wharton.result_bundle import verify_bundle

    typer.echo(json.dumps(verify_bundle(path)))


if __name__ == "__main__":
    app()
