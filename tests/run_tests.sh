#!/usr/bin/env bash
# -*- coding: utf-8 -*-

PYTHONPATH=.:./tests:$PYTHONPATH py.test -v tests --cov=scripts
