#!/bin/sh
source ~/.bash_profile
export DAY=20121214
hcat /projects/newsrtuf/exploredata_12_auto/ALL/$DAY/finalpvs.out/* | zcat  > $DAY.raw
