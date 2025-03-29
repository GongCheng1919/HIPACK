# make clean && 
make all

DEVICE="m1pro"
THREAD="mt"

# Test the correcteness of the HIPACK
# Too large input size will cause the baseline program to run for a long time
HIPACK_EXE=hipack_${THREAD}.exe
N=(1 2 4 8)
Ci=(32 64 128 256)
H=(8 16 32)
W=(8 16 32)
Co=(32 64 128 256)
WA_bits=(3 )
for n in "${N[@]}"; do
	for ci in "${Ci[@]}"; do
		for h in "${H[@]}"; do
			for w in "${W[@]}"; do
				for co in "${Co[@]}"; do
					for wa_bit in "${WA_bits[@]}"; do
						w_bit=${wa_bit}
						a_bit=${wa_bit}
						LOG_FILE=logs/test_hipack_correcteness_W${w_bit}A${a_bit}.log
						debug=1
						# If the input size is too large, we can set verbose=0
						size=$((n*ci*h*w))
						verbose=0
						if [ $size -gt 100 ]; then
							verbose=0
						fi
						echo "config: N${n} Ci${ci} H${h} W${w} Co${co} W${w_bit}A${a_bit} debug${debug} verbose${verbose}" >> ${LOG_FILE}
						# echo "config: N${n} Ci${ci} H${h} W${w} Co${co} W${w_bit}A${a_bit} debug${debug} verbose${verbose}" >> ${LOG_FILE}
						# ./${st_exe} ${n} ${ci} ${h} ${w} ${co} ${w_bit} ${a_bit} ${mt} | tee -a ${LOG_FILE}
						./${HIPACK_EXE} ${n} ${ci} ${h} ${w} ${co} ${w_bit} ${a_bit} ${debug} ${verbose} | tee -a ${LOG_FILE}
					done
				done
			done
		done
	done
done

# Test the performance of the HIPACK
N=16
H=224
W=224
C=32

for i in {1,2,3,4,5}; do
	for w_bit in "${WA_bits[@]}"; do
		a_bit=${w_bit}
		echo "config: W${w_bit}A${a_bit}"
		LOG_FILE=logs/test_hipack_perf_${DEVICE}_${THREAD}_W${w_bit}A${a_bit}.log
		mkdir -p logs

		echo "config: W${w_bit}A${a_bit}, save to: ${LOG_FILE}"
		./${HIPACK_EXE} ${N} 3 $((H)) $((W)) $((C)) ${w_bit} ${a_bit} 0 0  | tee ${LOG_FILE}
		./${HIPACK_EXE} ${N} $((C)) $((H/2)) $((W/2)) $((C*2)) ${w_bit} ${a_bit} 0 0  | tee -a ${LOG_FILE}
		./${HIPACK_EXE} ${N} $((C*2)) $((H/2)) $((W/2)) $((C*2)) ${w_bit} ${a_bit} 0 0  | tee -a ${LOG_FILE}
		./${HIPACK_EXE} ${N} $((C*2)) $((H/4)) $((W/4)) $((C*4)) ${w_bit} ${a_bit} 0 0 | tee -a ${LOG_FILE}
		./${HIPACK_EXE} ${N} $((C*4)) $((H/4)) $((W/4)) $((C*4)) ${w_bit} ${a_bit} 0 0 | tee -a ${LOG_FILE}
		./${HIPACK_EXE} ${N} $((C*4)) $((H/8)) $((W/8)) $((C*8)) ${w_bit} ${a_bit} 0 0 | tee -a ${LOG_FILE}
		./${HIPACK_EXE} ${N} $((C*8)) $((H/8)) $((W/8)) $((C*8)) ${w_bit} ${a_bit} 0 0 | tee -a ${LOG_FILE}
		./${HIPACK_EXE} ${N} $((C*8)) $((H/16)) $((W/16)) $((C*16)) ${w_bit} ${a_bit} 0 0 | tee -a ${LOG_FILE}
		./${HIPACK_EXE} ${N} $((C*16)) $((H/16)) $((W/16)) $((C*16)) ${w_bit} ${a_bit} 0 0 | tee -a ${LOG_FILE}
		./${HIPACK_EXE} ${N} $((C*16)) $((H/32)) $((W/32)) $((C*32)) ${w_bit} ${a_bit} 0 0 | tee -a ${LOG_FILE}
		./${HIPACK_EXE} ${N} $((C*32)) $((H/32)) $((W/32)) $((C*32)) ${w_bit} ${a_bit} 0 0 | tee -a ${LOG_FILE}
	done
done