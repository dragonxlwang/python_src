#!/bin/sh
source ~/.bash_profile
svn_update
python vwExampleGenerator.py < ~/data/20121214.raw 
