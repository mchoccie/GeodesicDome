#!/bin/bash

coverage run --source=. -m pytest && coverage report -m --omit="*/test*"