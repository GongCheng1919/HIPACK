DEVICE="rpi4_yolov5s"
HIPACK_EXE=usage_of_directconv_args.py

# Test the performance of the HIPACK
for i in $(seq 1 5); do
	w_bit=3
	a_bit=${w_bit}
	echo "config: W${w_bit}A${a_bit}"
	LOG_FILE=logs/test_hipack_perf_${DEVICE}_W${w_bit}A${a_bit}_${i}.log
	mkdir -p logs
	echo "config: W${w_bit}A${a_bit}, save to: ${LOG_FILE}"

	python3 ./${HIPACK_EXE} --N 32 --Ci 32 --H 160 --W 160 --Co 32 --W_bits ${w_bit} --A_bits ${a_bit}| tee ${LOG_FILE}
	python3 ./${HIPACK_EXE} --N 32 --Ci 64 --H 80 --W 80 --Co 64 --W_bits ${w_bit} --A_bits ${a_bit}| tee ${LOG_FILE}
	python3 ./${HIPACK_EXE} --N 32 --Ci 128 --H 40 --W 40 --Co 128 --W_bits ${w_bit} --A_bits ${a_bit}| tee ${LOG_FILE}
	python3 ./${HIPACK_EXE} --N 32 --Ci 256 --H 20 --W 20 --Co 256 --W_bits ${w_bit} --A_bits ${a_bit}| tee ${LOG_FILE}
done
