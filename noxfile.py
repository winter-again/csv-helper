import nox

nox.options.default_venv_backend = "uv"


@nox.session(python=["3.12", "3.13"])
def tests(session: nox.Session) -> None:
    """Run tests with pytest."""
    # TODO: run_install vs run? needs install step...
    session.run_install(
        "uv",
        "sync",
        "--locked",
        f"--python={session.virtualenv.location}",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    session.run("pytest", "-vv", "tests")
