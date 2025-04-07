# make clean && 
make all

DEVICE="rpi4_diffusion"
THREAD="mt"

# Test the correcteness of the HIPACK
# Too large input size will cause the baseline program to run for a long time
HIPACK_EXE=hipack_${THREAD}.exe
WA_bits=(3 4)


# Test the performance of the HIPACK


for i in {1,2,3,4,5}; do
	for w_bit in "${WA_bits[@]}"; do
		a_bit=${w_bit}
		echo "config: W${w_bit}A${a_bit}"
		LOG_FILE=logs/test_hipack_perf_${DEVICE}_${THREAD}_W${w_bit}A${a_bit}.log
		mkdir -p logs

		echo "config: W${w_bit}A${a_bit}, save to: ${LOG_FILE}"
		# ./${HIPACK_EXE} ${N} $((C)) $((H/2)) $((W/2)) $((C*2)) ${w_bit} ${a_bit} 0 0  | tee -a ${LOG_FILE}
		./${HIPACK_EXE} 32 32 160 160 32 0 0  | tee ${LOG_FILE}
		./${HIPACK_EXE} 32 64 80 80 64 0 0  | tee ${LOG_FILE}
		./${HIPACK_EXE} 32 128 40 40 128 0 0  | tee ${LOG_FILE}
		./${HIPACK_EXE} 32 256 20 20 256 0 0  | tee ${LOG_FILE}
	done
done