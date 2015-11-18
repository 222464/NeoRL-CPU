Welcome to NeoRL-CPU.

## Build

```
cd /path/to/NeoRL-CPU/NeoRL-CPU
cmake .
make -j4
```

## Run simple sequence learning experiment

```
cd /path/to/NeoRL-CPU/NeoRL-CPU
echo "Instead of relying on real world data, we can instead challenge the machine learning models" > phrase.txt
./NeoRL-CPU --epochs 100 --seed 1337 --corpus phrase.txt --samples 40 --nlayers 1 --lw 16 --lh 16 --ifbradius 12 --sprednoise 0.02 --sseednoise 0.4
```
