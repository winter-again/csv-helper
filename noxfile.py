import nox

nox.options.default_venv_backend = "uv"


@nox.session
def tests(session: nox.Session) -> None:
    """Run tests with pytest."""
    session.run_install(
        "uv",
        "sync",
        "--locked",
        f"--python={session.virtualenv.location}",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    session.run("pytest", "-vv", "tests")
