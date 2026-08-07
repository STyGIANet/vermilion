#!/bin/bash

for noise in {0..9};do 
	(python3 vermilion2.py $noise >/dev/null 2> /dev/null &);
done