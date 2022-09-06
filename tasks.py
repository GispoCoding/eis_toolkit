from invoke import task


@task
def lint(ctx):
    ctx.run("mypy eis_toolkit", pty=True)
    ctx.run("flake8 eis_toolkit", pty=True)
    ctx.run("black eis_toolkit", pty=True)
    ctx.run("isort eis_toolkit", pty=True)
