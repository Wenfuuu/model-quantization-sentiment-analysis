#!/bin/bash
git fetch origin main
git merge origin/main
python main.py
