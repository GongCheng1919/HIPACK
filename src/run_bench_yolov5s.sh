# make clean && 
make all

DEVICE="rpi4_yolov5s"
THREAD="mt"

# Test the correcteness of the HIPACK
# Too large input size will cause the baseline program to run for a long time
HIPACK_EXE=hipack_${THREAD}.exe
WA_bits=(3 4)


# Test the performance of the HIPACK


for i in {1,2,3,4,5}; do
	w_bit=3
	a_bit=${w_bit}
	echo "config: W${w_bit}A${a_bit}"
	LOG_FILE=logs/test_hipack_perf_${DEVICE}_${THREAD}_W${w_bit}A${a_bit}.log
	mkdir -p logs

	echo "config: W${w_bit}A${a_bit}, save to: ${LOG_FILE}"
	# ./${HIPACK_EXE} ${N} $((C)) $((H/2)) $((W/2)) $((C*2)) ${w_bit} ${a_bit} 0 0  | tee -a ${LOG_FILE}
	./${HIPACK_EXE} 32 3 56 56 64 ${w_bit} ${a_bit} 0 0  | tee ${LOG_FILE}
	./${HIPACK_EXE} 32 64 56 56 64 ${w_bit} ${a_bit} 0 0  | tee ${LOG_FILE}
	./${HIPACK_EXE} 32 64 28 28 64 ${w_bit} ${a_bit} 0 0  | tee ${LOG_FILE}
	./${HIPACK_EXE} 32 128 14 14 128 ${w_bit} ${a_bit} 0 0 | tee ${LOG_FILE}
	./${HIPACK_EXE} 32 256 7 7 256 ${w_bit} ${a_bit} 0 0 | tee ${LOG_FILE}
	./${HIPACK_EXE} 32 256 7 7 512 ${w_bit} ${a_bit} 0 0 | tee ${LOG_FILE}
	./${HIPACK_EXE} 32 512 7 7 512 ${w_bit} ${a_bit} 0 0 | tee ${LOG_FILE}
	./${HIPACK_EXE} 32 768 7 7 512 ${w_bit} ${a_bit} 0 0 | tee ${LOG_FILE}
	./${HIPACK_EXE} 32 512 14 14 256 ${w_bit} ${a_bit} 0 0 | tee ${LOG_FILE}
	./${HIPACK_EXE} 32 384 14 14 256 ${w_bit} ${a_bit} 0 0 | tee ${LOG_FILE}
	./${HIPACK_EXE} 32 256 14 14 256 ${w_bit} ${a_bit} 0 0 | tee ${LOG_FILE}
	./${HIPACK_EXE} 32 256 28 28 128 ${w_bit} ${a_bit} 0 0 | tee ${LOG_FILE}
	./${HIPACK_EXE} 32 192 28 28 128 ${w_bit} ${a_bit} 0 0 | tee ${LOG_FILE}
	./${HIPACK_EXE} 32 128 28 28 128 ${w_bit} ${a_bit} 0 0 | tee ${LOG_FILE}
	./${HIPACK_EXE} 32 128 56 56 64 ${w_bit} ${a_bit} 0 0 | tee ${LOG_FILE}
done