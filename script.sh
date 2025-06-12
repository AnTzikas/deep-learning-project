#!/bin/bash

#echo -e "\n\nRunning python ex_4.py --model NeuMF"
#python ex_4.py --model NeuMF

echo -e "\n\nRunning python ex_4.py --model GMF"
python ex_4.py --model GMF

echo -e "\n\nRunning python ex_4.py --model MLP"
python ex_4.py --model MLP

echo -e "\n\nRunning python ex_5.py"
python ex_5.py

echo -e "\n\nRunning python ex_7.py"
python ex_7.py
