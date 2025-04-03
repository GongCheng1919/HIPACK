MACHINE_NAME="aarch64"
g++ ./test.cpp -o ./test.exe -march=native -lcpuinfo -lclog -fconcepts-ts -std=c++17 -Ofast -I/usr/include/$MACHINE_NAME-linux-gnu/ -L/usr/include/$MACHINE_NAME-linux-gnu -lopenblas
./test.exe