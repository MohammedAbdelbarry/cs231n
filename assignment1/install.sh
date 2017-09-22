#!/bin/sh
for line in `cat requirements.txt`
do
   	sudo -H pip3 install $line
done
