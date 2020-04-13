<p align="center">
<a href="https://github.com/ambv/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

# hippo7
<center>"Why is this called hippo7? I don't know, but hippos are cool."</center>

hippo7 is a tool for visual live shows that can be synchronized to music.

The tool was build by [Benedikt Wiberg](http://github.com/qway/) and myself during an TUM Project at [Luminovo.ai](Luminovo.ai).

## Development

### Pre-commit
Please make sure to have the pre-commit hooks installed.
Install [pre-commit](https://pre-commit.com/) and then run `pre-commit install` to register the hooks with git.

### Poetry
Use [poetry](https://poetry.eustace.io/) to manage your dependencies.
Please make sure it is installed for your current python version.
Then start by adding your dependencies:
```console
poetry add torch
```

### Makefile
We use [make](https://www.gnu.org/software/make/) to streamline our development workflow.
Run `make help` to see all available commands.

<!-- START makefile-doc -->
```
$ make help 
help                 Show this help message
build                Build the docker container and docs
check                Run all static checks (like pre-commit hooks)
docs                 Serve all docs
test                 Run all tests
test-docker          Run tests in the docker environment
dev-install          Install all the packages in the local python environment for development
run-server           Runs the render server
run-client           Runs the client to control the server 
```
<!-- END makefile-doc -->
