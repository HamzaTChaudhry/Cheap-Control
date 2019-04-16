#!/bin/bash

pvpython animate_simulation.py FF_07x010x07-05 local 2 &
pvpython animate_simulation.py FF_07x020x07-27 local 2 &
pvpython animate_simulation.py FF_07x030x07-37 local 2 &
wait
pvpython animate_simulation.py FF_07x040x07-47 local 2 &
pvpython animate_simulation.py FF_07x050x07-09 local 2 &
pvpython animate_simulation.py FF_07x060x07-38 local 2 &
wait
pvpython animate_simulation.py FF_07x070x07-10 local 2 &
pvpython animate_simulation.py FF_07x080x07-26 local 2 &
pvpython animate_simulation.py FF_07x090x07-43 local 2 &
pvpython animate_simulation.py FF_07x100x07-40 local 2 &

# I need to figure out how to automatically read the model names from the CSV and cycle through them in a for loop in bash syntax.

# declare -a networks=("FF_03x010x03-10" "FF_03x020x03-35" "FF_03x030x03-01" "FF_03x040x03-45" "FF_03x050x03-28" "FF_03x060x03-27" "FF_03x070x03-10" "FF_03x080x03-41" "FF_03x090x03-19" "FF_03x100x03-10")

# for i in "${networks[@]}"
# do 
#     pvpython animate_simulation.py "$i" local 2 
# done