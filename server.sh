#!/bin/bash
cd "$(dirname "$0")"
.venv/bin/pip install -r requirements.txt
.venv/bin/python3 server.py
