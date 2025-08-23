#!/bin/bash -ex
export ALGORITHM=${ALGORITHM:-shapleyiq}
LOGURU_COLORIZE=0 .venv/bin/python container run
