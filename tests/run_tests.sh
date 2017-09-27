#!/usr/bin/env bash
# -*- coding: utf-8 -*-
rm -rf tests/__pycache__
rm -rf tests/.cache

PYTHONPATH=.:./tests:$PYTHONPATH py.test -v tests --cov=scripts
