# make clean && 
make all

DEVICE="rpi4_diffusion"
THREAD="mt"

# Test the correcteness of the HIPACK
# Too large input size will cause the baseline program to run for a long time
HIPACK_EXE=hipack_${THREAD}.exe


# Test the performance of the HIPACK


for i in $(seq 1 5); do
	w_bit=3
	a_bit=${w_bit}
	echo "config: W${w_bit}A${a_bit}"
	LOG_FILE=logs/test_hipack_perf_${DEVICE}_${THREAD}_W${w_bit}A${a_bit}_${i}.log
	mkdir -p logs
	echo "config: W${w_bit}A${a_bit}, save to: ${LOG_FILE}"
	# ./${HIPACK_EXE} ${N} $((C)) $((H/2)) $((W/2)) $((C*2)) ${w_bit} ${a_bit} 0 0  | tee -a -a ${LOG_FILE}
	./${HIPACK_EXE} 32 32 160 160 32 ${w_bit} ${a_bit} 0 0 | tee -a ${LOG_FILE}
	./${HIPACK_EXE} 32 64 80 80 64 ${w_bit} ${a_bit} 0 0   | tee -a ${LOG_FILE}
	./${HIPACK_EXE} 32 128 40 40 128 ${w_bit} ${a_bit} 0 0 | tee -a ${LOG_FILE}
	./${HIPACK_EXE} 32 256 20 20 256 ${w_bit} ${a_bit} 0 0 | tee -a ${LOG_FILE}
done
