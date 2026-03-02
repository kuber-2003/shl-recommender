#!/bin/bash
pip install -r requirements.txt
cp scraper/data/shl_assessments.json data/shl_assessments.json
cd backend && python build_index.py