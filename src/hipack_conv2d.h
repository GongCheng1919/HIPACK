#include "base.h"

void hipack_conv2d(const int *inp_ptr, const int *weight_ptr, float *output_ptr,
				   int N, int Ci, int H, int W, int Co, int K, int Ho, int Wo, int padding = 0,
				   int a_bit = 4, int w_bit = 2, int ar_bit = 32, int wr_bit = 32,
				   int stride = 1, int dilation = 1)
{

	// test_convert_uint8x16_to_uint16x8_nbits();
	// 1, 第一个改进点，将guard_bit从12比特提升到14bit，就是为了能够在最内层循环上累加更多次，从而尽可能地延迟解包计算的时间
	// 但是当我们假设最后一个数只需要6bit存储时，其实相当于58bit存储4个数，因此每个数最大可以用58/4=14bit存储
	// const int guard_bit = 14; // using the 8-bit to split the results as uchar.
	const int guard_bit = 13; // (x-2)+4x<=64 ==> x=13
	const int align_bits = 8;
	const int TN = 2;
	const int regN = 4;
	const int regW = 8;
	const int regA = 2;
	const int exbits = 1;
	// 每个weight压缩到uint8，三个weight存储到uint32,并将weight的元素进行顺序存储
	int *packed_weights;
	posix_memalign((void **)&packed_weights, align_bits, Co * K * Ci * align_bits);
	for (int y = 0; y < Co; y += regW)
	{
		const int y_upper = min(regW, Co - y);
		for (int kh = 0; kh < K; kh++)
		{
			for (int i = 0; i < Ci; i++)
			{
				for (int yy = 0; yy < y_upper; yy++)
				{
					// packed_weights[y*Ci*K+i*regW*K+yy*K+kh]=0; // (y/regW)*Ci*regW+i*regW+yy
					int weight_idx = (y + yy) * Ci * K * K + i * K * K + kh * K;
					int packed_weight_idx = y * K * Ci + kh * Ci * regW + i * regW + yy; // (y/regW)*Ci*regW+i*regW+yy
					packed_weights[packed_weight_idx] = 0;
					for (int k = 0; k < K; k++)
					{
						packed_weights[packed_weight_idx] += (unsigned int)weight_ptr[weight_idx++] << (guard_bit * (K - k - 1));
					}
				}
			}
		}
	}

	// packing inputs: 在分块后可以使用常量小数组加速
	int d_Wo = Wo / K;
	int *packed_inputs;
	posix_memalign((void **)&packed_inputs, align_bits, N * Ci * H * d_Wo * align_bits);
	// 原始的inputs的内存顺序为[N*Ci*H*W]
	// 重新将packed_inputs排序为[H*N*Ci*d_Wo]=》
	//     这样重排仍然会导致问题，那就是当将Wo按RegA*TN进行分块后，每一次读取下一个分块的时候会跨过不连续的距离导致跳跃访问，从而导致极大的访存性能开销
	// for (int x=0;x<N;x++){
	//     for (int i=0;i<Ci;i++){
	//         for(int h=0;h<H;h++){
	//             for (int z=0;z<Wo;z+=K){
	//                 // 原始的inputs的内存顺序为[N*Ci*H*W]
	//                 // int idx = x*Ci*H*d_Wo+i*H*d_Wo+h*d_Wo+z/K;
	//                 // 重新将packed_inputs排序为[H*N*Ci*d_Wo]
	//                 int idx = h*N*Ci*d_Wo+x*Ci*d_Wo+i*d_Wo+z/K;
	//                 packed_inputs[idx] = 0;
	//                 for (int k=0;k<K;k++){
	//                     int pos = int(z*stride+k*dilation);
	//                     int inp_idx = x*Ci*H*W+i*H*W+h*W+pos;
	//                     if (pos<W)
	//                         packed_inputs[idx] += (unsigned int)inp_ptr[inp_idx]<<(guard_bit*k);
	//                 }
	//             }
	//         }
	//     }
	// }
	// 做第二次重排：
	for (int x = 0; x < N; x += regN)
	{
		const int x_upper = min(regN, N - x);
		const int x_len = N / regN;
		int x_iter = x / regN;
		for (int h = 0; h < H; h++)
		{
			for (int z = 0; z <= d_Wo; z += regA * TN)
			{ // K is the interval: 0,3,6 // <=d_Wo-TN is to
				const int z_upper = min(regA * TN, d_Wo - z);
				const int z_len = d_Wo / (regA * TN);
				int z_iter = (z / (regA * TN));

				for (int i = 0; i < Ci; i++)
				{
					for (int xx = 0; xx < x_upper; xx++)
					{
						for (int zz = 0; zz < z_upper; zz++)
						{
							// 原始的inputs的内存顺序为[N*Ci*H*W]
							// int idx = x*Ci*H*d_Wo+i*H*d_Wo+h*d_Wo+z/K;
							// 重新将packed_inputs排序为[H*N*Ci*d_Wo]
							// int idx = h*N*Ci*d_Wo+(x+xx)*Ci*d_Wo+i*d_Wo+(z+zz);
							// 设置这个是一个6维[N/regN, Ho, d_Wo/(regA*TN),Ci,x_upper,z_upper]的数组而不是4维的，否则访存永远不连续因为你的id就不连续
							int idx = x_iter * (H * z_len * Ci * x_upper * z_upper) +
									  h * (z_len * Ci * x_upper * z_upper) +
									  z_iter * (Ci * x_upper * z_upper) +
									  i * x_upper * z_upper +
									  xx * z_upper +
									  zz;
							// printf("[x=%d,h=%d,z=%d,i=%d,xx=%d,zz=%d, z_upper=%d]",(x/regN),h,(z/(regA*TN)),i,xx,zz,z_upper);
							// std::cout<<"input_idx:"<<idx<<std::endl;
							int raw_z = (z + zz) * K;
							packed_inputs[idx] = 0;
							for (int k = 0; k < K; k++)
							{
								int pos = int(raw_z * stride + k * dilation);
								int inp_idx = (x + xx) * Ci * H * W + i * H * W + h * W + pos;
								if (pos < W)
									packed_inputs[idx] += (unsigned int)inp_ptr[inp_idx] << (guard_bit * k);
							}
						}
					}
				}
			}
		}
	}

	int micro_iter_num = Ci;
	int max_acc_3_w_mul_a = ((w_bit + a_bit) + 2);
	if (w_bit == 1 || a_bit == 1)
	{
		max_acc_3_w_mul_a -= 1;
	}
	int micro_u8_iter_num = min(micro_iter_num, 2 * pow(2, exbits + (guard_bit - max_acc_3_w_mul_a))); // uint8缓存n次中间结果

	uint64_t mask = ((1ll << guard_bit) - 1);
	uint64_t low_mask = (mask << (guard_bit)) | (mask << (3 * guard_bit));
	uint64_t high_mask = mask | (mask << (2 * guard_bit)) | (mask << (4 * guard_bit));
	uint64x2_t low_maskx2 = {low_mask, low_mask};
	uint64x2_t high_maskx2 = {high_mask, high_mask};

	// Test_accumlate_uint64x2x2_to_8xfloat32_nbits();
	// exit(1);
	// 	omp_set_num_threads(8);
	// #pragma omp parallel for collapse(3) // 展开三重循环
	for (int x = 0; x < N; x += regN)
	{
		// #pragma omp parallel for
		for (int y = 0; y < Co; y += regW)
		{
			// #pragma omp parallel for
			// int H = H;
			// for (int h = -(K - 1); h < H; h++)
			for (int h = 0; h < H; h++)
			{
				for (int z = 0; z <= d_Wo; z += regA * TN)
				{ // K is the interval: 0,3,6 // <=d_Wo-TN is to
					int kh_low = (h < 0) * (-h);
					int kh_upper = (H - h > K) ? K : (H - h);
					const int y_upper = min(regW, Co - y);
					const int x_upper = min(regN, N - x);
					int x_iter = x / regN;
					const int z_upper = min(regA * TN, d_Wo - z);
					const int z_len = d_Wo / (regA * TN);
					int z_iter = (z / (regA * TN));
					// 取消这个缓冲来节约寄存器资源
					// float32_4 output_cache_vec[regN*regW*regA*2] = {0};
					uint64x2x2_t dual_local_accumulator[regN * regW * regA] = {0};

					// kh加在这里，因为kh的计算不能在int16和int8上做累加，会溢出，但是可以在float上累加，因此放在这里
					for (int kh = kh_low; kh < kh_upper; kh++)
					{
						int posh = int(h * stride + kh * dilation);
						// if (posh>=H){
						//     continue;
						// }
						for (int i = 0; i < Ci; i += micro_iter_num)
						{
							// uint16_8 micro_u16_output_cache[regN*regW*regA] = {0}; // 用于缓存256次累加结果
							const int i_upper = min(micro_iter_num, Ci - i);

							for (int ii = 0; ii < i_upper; ii += micro_u8_iter_num)
							{
								const int ii_upper = min(micro_u8_iter_num, i_upper - ii);
								uint64_2 micro_u8_output_cache[regN * regW * regA] = {0}; // 用于缓存2次中间累加结果 // 不能缓存4次中间结果了,因为w2a4的结果还剩2bit刚好用来累加kernel的结果了
								for (int iii = 0; iii < ii_upper; ++iii)
								{
									// 2, 第二个改进点，将input和weight的循环次序做了调换，内层循环上load weight而不是input，从而尽可能地复用input
									// 由于input需要一次load2个int32而weight仅仅需要load一个int32，可以降低内存读取？cache miss呢？
									// #pragma GCC unroll 4
									for (int xx = 0; xx < x_upper; xx++)
									{
										for (int zz = 0; zz < z_upper; zz += TN)
										{
											// 这里的内存地址不连续，多了一个常数posh*d_Wo，意味着每一轮的循环的读取，都会跳过H*d_Wo+posh*d_Wo这么长一个空位，
											// 假设H是14，d_Wo=Wo/K=4, posh=h+kh=13+2=15,那么每一次空位长度应该为(14*4+15*4)*4B=464字节，
											// 也就是说每一次读取的时候，都会跳过464字节，而L1的cache line是64字节，也就是说每次读取的时候，都会有7个cache line miss
											// L1 的cache size是32KB，也就是说每一轮iii的循环，会跳过464字节，而内存中读取的长度为regA*TN*regW*4B=32B，
											// 也就是1轮循环会跳过464+32=496字节，也就是说，每一轮的循环，会有7*496/64=56个cache line miss，
											// 每过32KB/496B=64次循环就会有一个L1的cache miss， 而我们的iii循环是2*2^(14-8+1)=256次，也就是说，每一轮的循环，会有4次L1的cache miss
											// 原始的packed_inputs内存顺序为[N*Ci*H*d_Wo]
											// int input_idx = (x+xx)*Ci*H*d_Wo+(i+ii+iii)*H*d_Wo+posh*d_Wo+z+zz;
											// 重排后packed_inputs内存顺序为[H*N*Ci*d_Wo]：从28G提升到36G
											// 重排后还是有问题：
											//     比如z=0时，每一次i循环，假设i:0->1, zz:0->2->4,真实坐标变换是（d_Wo = 28/3=10）, 0*10+0+4=4 ->1*10+0+0 = 10
											//     比如z=1时，每一次i循环，假设i:0->1, zz:0->2->4,真实坐标变换是（d_Wo = 28/3=10）, 0*10+1+4=5 ->1*10+1+0 = 11
											// 这样在内层循环中每一次都会跳过6个进行访问而不是连续的，这可能在Wo很大的时候导致极大的访存问题（跳跃访存的问题）。增加regA会部分解决该问题，但是还是无法处理大输出的问题。
											// int input_idx = posh*N*Ci*d_Wo+(x+xx)*Ci*d_Wo+(i+ii+iii)*d_Wo+(z+zz);
											int input_idx = x_iter * (H * z_len * Ci * x_upper * z_upper) +
															posh * (z_len * Ci * x_upper * z_upper) +
															z_iter * (Ci * x_upper * z_upper) +
															i * x_upper * z_upper +
															xx * z_upper +
															zz;
											// std::cout<<"input_idx:"<<input_idx<<std::endl;
											uint32_2 input = load_uint32_2(packed_inputs + input_idx);
											// uint32_2 input = load_uint32_2(packed_inputs); // 这里测试如果input不存在cache miss的性能会是多少，实测可以提升3-4G左右

											for (int yy = 0; yy < y_upper; yy++)
											{
												// loading weight
												int packed_weight_idx = y * K * Ci + kh * Ci * regW + (i + ii + iii) * regW + yy; // (y/regW)*Ci*regW+i*regW+yy
												// std::cout<<"weight_idx:"<<packed_weight_idx<<std::endl;
												uint32_2 weight = load_broad_uint32_2(packed_weights[packed_weight_idx]);
												// uint32_2 weight = load_broad_uint32_2(packed_weights[0]);// 这里测试如果weight不存在cache miss的性能会是多少，实测几乎没有提升

												int cache_indx = xx * regW * regA + yy * regA + zz / TN;

												uint64_2 output = mul_low_uint32_to_uint64_(input, weight);

												// // 将10个uint8的计算结果先累加2次, 如果位宽够低还可以做更多次累加
												micro_u8_output_cache[cache_indx] = vaddq_u64(micro_u8_output_cache[cache_indx], output);
											}
										}
									}
								}
								// 将在guardbits上累加的值再次累加到2*guardbits的内存中，需要用dual向量寄存器
								for (int xx = 0; xx < x_upper; xx++)
								{
									for (int yy = 0; yy < y_upper; yy++)
									{
										for (int zz = 0; zz < z_upper; zz += TN)
										{
											int cache_indx = xx * regW * regA + yy * regA + zz / TN;
											// int output_idx = (x+xx)*Co*Ho*Wo+(y+yy)*Ho*Wo+h*Wo+(z+zz)*K;
											// 然后将10个uint12的累加结果转换为8个uint16的进行最多256次累加
											// uint16_8 output_uint16x2 = convert_uint8x16_to_uint16x8(vreinterpretq_u8_u64(micro_u8_output_cache[cache_indx]));
											// uint16_8 output_uint16x2;
											// convert_uint8x16_to_uint16x8_nbits(micro_u8_output_cache[cache_indx],output_uint16x2, guard_bit);
											// accumlate_uint16x8_to_uint16x8(output_uint16x2,micro_u16_output_cache[cache_indx]);
											// float32_4 low_output = vcvtq_f32_u32(vmovl_u16(vget_low_u16(output_uint16x2)));
											// float32_4 high_output = vcvtq_f32_u32(vmovl_u16(vget_high_u16(output_uint16x2)));
											// AddFloat4(&output_ptr[output_idx+0], low_output);
											// AddFloat4(&output_ptr[output_idx+4], high_output);
											// accumlate_uint64x2_to_8xfloat32_nbits(micro_u8_output_cache[cache_indx],&output_ptr[output_idx], guard_bit);
											// 这一步缓存在W6A6条件下提升了大约1倍的计算性能：4G->11G
											dual_local_accumulator[cache_indx].val[0] = vaddq_u64(dual_local_accumulator[cache_indx].val[0],
																								  vandq_u64(micro_u8_output_cache[cache_indx], high_maskx2));
											dual_local_accumulator[cache_indx].val[1] = vaddq_u64(dual_local_accumulator[cache_indx].val[1],
																								  vandq_u64(micro_u8_output_cache[cache_indx], low_maskx2));
										}
									}
								}
							}
							// 最后将累加结果转换为float32
							for (int xx = 0; xx < x_upper; xx++)
							{
								for (int yy = 0; yy < y_upper; yy++)
								{
									for (int zz = 0; zz < z_upper; zz += TN)
									{
										int cache_indx = xx * regW * regA + yy * regA + zz / TN;
										int output_idx = (x + xx) * Co * Ho * Wo + (y + yy) * Ho * Wo + (h)*Wo + (z + zz) * K; // output_idx is decided by x,y,z. irrelated to i
										// float32_4 low_output = vcvtq_f32_u32(vmovl_u16(vget_low_u16(micro_u16_output_cache[cache_indx])));
										// float32_4 high_output = vcvtq_f32_u32(vmovl_u16(vget_high_u16(micro_u16_output_cache[cache_indx])));
										// AddFloat4(&output_ptr[output_idx+0], low_output);
										// AddFloat4(&output_ptr[output_idx+4], high_output);
										// float32_4* target_vec = output_cache_vec+(yy)*regA*2+zz/TN*2;
										// accumlate_uint16x8_to_2float32x4(micro_u16_output_cache[cache_indx],target_vec);
										accumlate_uint64x2x2_to_8xfloat32_nbits(dual_local_accumulator[cache_indx], &output_ptr[output_idx], guard_bit);
									}
								}
							}
						}
					}

					// 累加到output上
					// for (int xx = 0; xx<x_upper; xx++){
					//     for(int yy =0;yy<y_upper;yy++){
					//         for(int zz=0;zz<z_upper;zz+=TN){
					//             int output_idx = (x+xx)*Co*Ho*Wo+(y+yy)*Ho*Wo+h*Wo+(z+zz)*K; // output_idx is decided by x,y,z. irrelated to i
					//             float32_4* target_vec = output_cache_vec+(yy)*regA*2+zz/TN*2;
					//             AddFloat4(&output_ptr[output_idx+0], target_vec[0]);
					//             AddFloat4(&output_ptr[output_idx+4], target_vec[1]);
					//         }
					//     }
					// }
				}
			}
		}
	}
	std::free(packed_weights);
	std::free(packed_inputs);
}

void hipack_conv2d_v2(const int *inp_ptr, const int *weight_ptr, float *output_ptr,
					  int N, int Ci, int H, int W, int Co, int K, int Ho, int Wo, int padding = 0,
					  int a_bit = 4, int w_bit = 2, int ar_bit = 32, int wr_bit = 32,
					  int stride = 1, int dilation = 1)
{

	// test_convert_uint8x16_to_uint16x8_nbits();
	// 1, 第一个改进点，将guard_bit从12比特提升到14bit，就是为了能够在最内层循环上累加更多次，从而尽可能地延迟解包计算的时间
	// 但是当我们假设最后一个数只需要6bit存储时，其实相当于58bit存储4个数，因此每个数最大可以用58/4=14bit存储
	// const int guard_bit = 14; // using the 8-bit to split the results as uchar.
	const int guard_bit = 13; // (x-2)+4x<=64 ==> x=13
	const int align_bits = 8;
	const int regN = 4;
	const int regW = 8;
	const int regA = 2;
	const int exbits = 1;
	const int TN = 2;
	// 每个weight压缩到uint8，三个weight存储到uint32,并将weight的元素进行顺序存储
	int *packed_weights;
	posix_memalign((void **)&packed_weights, align_bits, Co * K * Ci * align_bits);
	for (int y = 0; y < Co; y += regW)
	{
		const int y_upper = min(regW, Co - y);
		for (int kh = 0; kh < K; kh++)
		{
			for (int i = 0; i < Ci; i++)
			{
				for (int yy = 0; yy < y_upper; yy++)
				{
					// packed_weights[y*Ci*K+i*regW*K+yy*K+kh]=0; // (y/regW)*Ci*regW+i*regW+yy
					int weight_idx = (y + yy) * Ci * K * K + i * K * K + kh * K;
					int packed_weight_idx = y * K * Ci + kh * Ci * regW + i * regW + yy; // (y/regW)*Ci*regW+i*regW+yy
					packed_weights[packed_weight_idx] = 0;
					for (int k = 0; k < K; k++)
					{
						packed_weights[packed_weight_idx] += (unsigned int)weight_ptr[weight_idx++] << (guard_bit * (K - k - 1));
					}
				}
			}
		}
	}

	// packing inputs: 在分块后可以使用常量小数组加速
	int d_Wo = Wo / K;
	int *packed_inputs;
	posix_memalign((void **)&packed_inputs, align_bits, N * Ci * H * d_Wo * align_bits);
	// 原始的inputs的内存顺序为[N*Ci*H*W]
	// 重新将packed_inputs排序为[H*N*Ci*d_Wo]=》
	//     这样重排仍然会导致问题，那就是当将Wo按RegA*TN进行分块后，每一次读取下一个分块的时候会跨过不连续的距离导致跳跃访问，从而导致极大的访存性能开销
	// for (int x=0;x<N;x++){
	//     for (int i=0;i<Ci;i++){
	//         for(int h=0;h<H;h++){
	//             for (int z=0;z<Wo;z+=K){
	//                 // 原始的inputs的内存顺序为[N*Ci*H*W]
	//                 // int idx = x*Ci*H*d_Wo+i*H*d_Wo+h*d_Wo+z/K;
	//                 // 重新将packed_inputs排序为[H*N*Ci*d_Wo]
	//                 int idx = h*N*Ci*d_Wo+x*Ci*d_Wo+i*d_Wo+z/K;
	//                 packed_inputs[idx] = 0;
	//                 for (int k=0;k<K;k++){
	//                     int pos = int(z*stride+k*dilation);
	//                     int inp_idx = x*Ci*H*W+i*H*W+h*W+pos;
	//                     if (pos<W)
	//                         packed_inputs[idx] += (unsigned int)inp_ptr[inp_idx]<<(guard_bit*k);
	//                 }
	//             }
	//         }
	//     }
	// }
	// 做第二次重排：
	for (int x = 0; x < N; x += regN)
	{
		const int x_upper = min(regN, N - x);
		const int x_len = N / regN;
		int x_iter = x / regN;
		for (int h = 0; h < H; h++)
		{
			for (int z = 0; z <= d_Wo; z += regA * TN)
			{ // K is the interval: 0,3,6 // <=d_Wo-TN is to
				const int z_upper = min(regA * TN, d_Wo - z);
				const int z_len = d_Wo / (regA * TN);
				int z_iter = (z / (regA * TN));

				for (int i = 0; i < Ci; i++)
				{
					for (int xx = 0; xx < x_upper; xx++)
					{
						for (int zz = 0; zz < z_upper; zz++)
						{
							// 原始的inputs的内存顺序为[N*Ci*H*W]
							// int idx = x*Ci*H*d_Wo+i*H*d_Wo+h*d_Wo+z/K;
							// 重新将packed_inputs排序为[H*N*Ci*d_Wo]
							// int idx = h*N*Ci*d_Wo+(x+xx)*Ci*d_Wo+i*d_Wo+(z+zz);
							// 设置这个是一个6维[N/regN, Ho, d_Wo/(regA*TN),Ci,x_upper,z_upper]的数组而不是4维的，否则访存永远不连续因为你的id就不连续
							int idx = x_iter * (H * z_len * Ci * x_upper * z_upper) +
									  h * (z_len * Ci * x_upper * z_upper) +
									  z_iter * (Ci * x_upper * z_upper) +
									  i * x_upper * z_upper +
									  xx * z_upper +
									  zz;
							// printf("[x=%d,h=%d,z=%d,i=%d,xx=%d,zz=%d, z_upper=%d]",(x/regN),h,(z/(regA*TN)),i,xx,zz,z_upper);
							// std::cout<<"input_idx:"<<idx<<std::endl;
							int raw_z = (z + zz) * K;
							packed_inputs[idx] = 0;
							for (int k = 0; k < K; k++)
							{
								int pos = int(raw_z * stride + k * dilation);
								int inp_idx = (x + xx) * Ci * H * W + i * H * W + h * W + pos;
								if (pos < W)
									packed_inputs[idx] += (unsigned int)inp_ptr[inp_idx] << (guard_bit * k);
							}
						}
					}
				}
			}
		}
	}

	int micro_iter_num = Ci;
	int max_acc_3_w_mul_a = ((w_bit + a_bit) + 2);
	if (w_bit == 1 || a_bit == 1)
	{
		max_acc_3_w_mul_a -= 1;
	}
	int micro_u8_iter_num = min(micro_iter_num, 2 * pow(2, exbits + (guard_bit - max_acc_3_w_mul_a))); // uint8缓存n次中间结果

	uint64_t mask = ((1ll << guard_bit) - 1);
	uint64_t low_mask = (mask << (guard_bit)) | (mask << (3 * guard_bit));
	uint64_t high_mask = mask | (mask << (2 * guard_bit)) | (mask << (4 * guard_bit));
	uint64x2_t low_maskx2 = {low_mask, low_mask};
	uint64x2_t high_maskx2 = {high_mask, high_mask};

	// Test_accumlate_uint64x2x2_to_8xfloat32_nbits();
	// exit(1);
	// 	omp_set_num_threads(8);
	// #pragma omp parallel for collapse(3) // 展开三重循环
	for (int x = 0; x < N; x += regN)
	{
		// #pragma omp parallel for
		for (int y = 0; y < Co; y += regW)
		{
			// #pragma omp parallel for
			// int H = H;
			for (int h = -(K - 1); h < H; h++)
			{
				for (int z = 0; z <= d_Wo; z += regA * TN)
				{ // K is the interval: 0,3,6 // <=d_Wo-TN is to
					int kh_low = (h < 0) * (-h);
					int kh_upper = (H - h > K) ? K : (H - h);
					const int y_upper = min(regW, Co - y);
					const int x_upper = min(regN, N - x);
					int x_iter = x / regN;
					const int z_upper = min(regA * TN, d_Wo - z);
					const int z_len = d_Wo / (regA * TN);
					int z_iter = (z / (regA * TN));
					// 取消这个缓冲来节约寄存器资源
					// float32_4 output_cache_vec[regN*regW*regA*2] = {0};
					uint64x2x2_t dual_local_accumulator[regN * regW * regA] = {0};

					// kh加在这里，因为kh的计算不能在int16和int8上做累加，会溢出，但是可以在float上累加，因此放在这里
					for (int kh = kh_low; kh < kh_upper; kh++)
					{
						int posh = int(h * stride + kh * dilation);
						// if (posh>=H){
						//     continue;
						// }
						for (int i = 0; i < Ci; i += micro_iter_num)
						{
							// uint16_8 micro_u16_output_cache[regN*regW*regA] = {0}; // 用于缓存256次累加结果
							const int i_upper = min(micro_iter_num, Ci - i);

							for (int ii = 0; ii < i_upper; ii += micro_u8_iter_num)
							{
								const int ii_upper = min(micro_u8_iter_num, i_upper - ii);
								uint64_2 micro_u8_output_cache[regN * regW * regA] = {0}; // 用于缓存2次中间累加结果 // 不能缓存4次中间结果了,因为w2a4的结果还剩2bit刚好用来累加kernel的结果了
								for (int iii = 0; iii < ii_upper; ++iii)
								{
									// 2, 第二个改进点，将input和weight的循环次序做了调换，内层循环上load weight而不是input，从而尽可能地复用input
									// 由于input需要一次load2个int32而weight仅仅需要load一个int32，可以降低内存读取？cache miss呢？
									// #pragma GCC unroll 4
									for (int xx = 0; xx < x_upper; xx++)
									{
										for (int zz = 0; zz < z_upper; zz += TN)
										{
											// 这里的内存地址不连续，多了一个常数posh*d_Wo，意味着每一轮的循环的读取，都会跳过H*d_Wo+posh*d_Wo这么长一个空位，
											// 假设H是14，d_Wo=Wo/K=4, posh=h+kh=13+2=15,那么每一次空位长度应该为(14*4+15*4)*4B=464字节，
											// 也就是说每一次读取的时候，都会跳过464字节，而L1的cache line是64字节，也就是说每次读取的时候，都会有7个cache line miss
											// L1 的cache size是32KB，也就是说每一轮iii的循环，会跳过464字节，而内存中读取的长度为regA*TN*regW*4B=32B，
											// 也就是1轮循环会跳过464+32=496字节，也就是说，每一轮的循环，会有7*496/64=56个cache line miss，
											// 每过32KB/496B=64次循环就会有一个L1的cache miss， 而我们的iii循环是2*2^(14-8+1)=256次，也就是说，每一轮的循环，会有4次L1的cache miss
											// 原始的packed_inputs内存顺序为[N*Ci*H*d_Wo]
											// int input_idx = (x+xx)*Ci*H*d_Wo+(i+ii+iii)*H*d_Wo+posh*d_Wo+z+zz;
											// 重排后packed_inputs内存顺序为[H*N*Ci*d_Wo]：从28G提升到36G
											// 重排后还是有问题：
											//     比如z=0时，每一次i循环，假设i:0->1, zz:0->2->4,真实坐标变换是（d_Wo = 28/3=10）, 0*10+0+4=4 ->1*10+0+0 = 10
											//     比如z=1时，每一次i循环，假设i:0->1, zz:0->2->4,真实坐标变换是（d_Wo = 28/3=10）, 0*10+1+4=5 ->1*10+1+0 = 11
											// 这样在内层循环中每一次都会跳过6个进行访问而不是连续的，这可能在Wo很大的时候导致极大的访存问题（跳跃访存的问题）。增加regA会部分解决该问题，但是还是无法处理大输出的问题。
											// int input_idx = posh*N*Ci*d_Wo+(x+xx)*Ci*d_Wo+(i+ii+iii)*d_Wo+(z+zz);
											int input_idx = x_iter * (H * z_len * Ci * x_upper * z_upper) +
															posh * (z_len * Ci * x_upper * z_upper) +
															z_iter * (Ci * x_upper * z_upper) +
															i * x_upper * z_upper +
															xx * z_upper +
															zz;
											// std::cout<<"input_idx:"<<input_idx<<std::endl;
											uint32_2 input = load_uint32_2(packed_inputs + input_idx);
											// uint32_2 input = load_uint32_2(packed_inputs); // 这里测试如果input不存在cache miss的性能会是多少，实测可以提升3-4G左右

											for (int yy = 0; yy < y_upper; yy++)
											{
												// loading weight
												int packed_weight_idx = y * K * Ci + kh * Ci * regW + (i + ii + iii) * regW + yy; // (y/regW)*Ci*regW+i*regW+yy
												// std::cout<<"weight_idx:"<<packed_weight_idx<<std::endl;
												uint32_2 weight = load_broad_uint32_2(packed_weights[packed_weight_idx]);
												// uint32_2 weight = load_broad_uint32_2(packed_weights[0]);// 这里测试如果weight不存在cache miss的性能会是多少，实测几乎没有提升

												int cache_indx = xx * regW * regA + yy * regA + zz / TN;

												uint64_2 output = mul_low_uint32_to_uint64_(input, weight);

												// // 将10个uint8的计算结果先累加2次, 如果位宽够低还可以做更多次累加
												micro_u8_output_cache[cache_indx] = vaddq_u64(micro_u8_output_cache[cache_indx], output);
											}
										}
									}
								}
								// 将在guardbits上累加的值再次累加到2*guardbits的内存中，需要用dual向量寄存器
								for (int xx = 0; xx < x_upper; xx++)
								{
									for (int yy = 0; yy < y_upper; yy++)
									{
										for (int zz = 0; zz < z_upper; zz += TN)
										{
											int cache_indx = xx * regW * regA + yy * regA + zz / TN;
											// int output_idx = (x+xx)*Co*Ho*Wo+(y+yy)*Ho*Wo+h*Wo+(z+zz)*K;
											// 然后将10个uint12的累加结果转换为8个uint16的进行最多256次累加
											// uint16_8 output_uint16x2 = convert_uint8x16_to_uint16x8(vreinterpretq_u8_u64(micro_u8_output_cache[cache_indx]));
											// uint16_8 output_uint16x2;
											// convert_uint8x16_to_uint16x8_nbits(micro_u8_output_cache[cache_indx],output_uint16x2, guard_bit);
											// accumlate_uint16x8_to_uint16x8(output_uint16x2,micro_u16_output_cache[cache_indx]);
											// float32_4 low_output = vcvtq_f32_u32(vmovl_u16(vget_low_u16(output_uint16x2)));
											// float32_4 high_output = vcvtq_f32_u32(vmovl_u16(vget_high_u16(output_uint16x2)));
											// AddFloat4(&output_ptr[output_idx+0], low_output);
											// AddFloat4(&output_ptr[output_idx+4], high_output);
											// accumlate_uint64x2_to_8xfloat32_nbits(micro_u8_output_cache[cache_indx],&output_ptr[output_idx], guard_bit);
											// 这一步缓存在W6A6条件下提升了大约1倍的计算性能：4G->11G
											dual_local_accumulator[cache_indx].val[0] = vaddq_u64(dual_local_accumulator[cache_indx].val[0],
																								  vandq_u64(micro_u8_output_cache[cache_indx], high_maskx2));
											dual_local_accumulator[cache_indx].val[1] = vaddq_u64(dual_local_accumulator[cache_indx].val[1],
																								  vandq_u64(micro_u8_output_cache[cache_indx], low_maskx2));
										}
									}
								}
							}
							// 最后将累加结果转换为float32
							for (int xx = 0; xx < x_upper; xx++)
							{
								for (int yy = 0; yy < y_upper; yy++)
								{
									for (int zz = 0; zz < z_upper; zz += TN)
									{
										int cache_indx = xx * regW * regA + yy * regA + zz / TN;
										int output_idx = (x + xx) * Co * Ho * Wo + (y + yy) * Ho * Wo + (h + (K - 1)) * Wo + (z + zz) * K; // output_idx is decided by x,y,z. irrelated to i
										// float32_4 low_output = vcvtq_f32_u32(vmovl_u16(vget_low_u16(micro_u16_output_cache[cache_indx])));
										// float32_4 high_output = vcvtq_f32_u32(vmovl_u16(vget_high_u16(micro_u16_output_cache[cache_indx])));
										// AddFloat4(&output_ptr[output_idx+0], low_output);
										// AddFloat4(&output_ptr[output_idx+4], high_output);
										// float32_4* target_vec = output_cache_vec+(yy)*regA*2+zz/TN*2;
										// accumlate_uint16x8_to_2float32x4(micro_u16_output_cache[cache_indx],target_vec);
										accumlate_uint64x2x2_to_8xfloat32_nbits(dual_local_accumulator[cache_indx], &output_ptr[output_idx], guard_bit);
									}
								}
							}
						}
					}

					// 累加到output上
					// for (int xx = 0; xx<x_upper; xx++){
					//     for(int yy =0;yy<y_upper;yy++){
					//         for(int zz=0;zz<z_upper;zz+=TN){
					//             int output_idx = (x+xx)*Co*Ho*Wo+(y+yy)*Ho*Wo+h*Wo+(z+zz)*K; // output_idx is decided by x,y,z. irrelated to i
					//             float32_4* target_vec = output_cache_vec+(yy)*regA*2+zz/TN*2;
					//             AddFloat4(&output_ptr[output_idx+0], target_vec[0]);
					//             AddFloat4(&output_ptr[output_idx+4], target_vec[1]);
					//         }
					//     }
					// }
				}
			}
		}
	}
	std::free(packed_weights);
	std::free(packed_inputs);
}

void hipack_conv2d_v3(const int *inp_ptr, const int *weight_ptr, float *output_ptr,
					  int N, int Ci, int H, int W, int Co, int K, int Ho, int Wo, int padding = 0,
					  int a_bit = 4, int w_bit = 2, int ar_bit = 32, int wr_bit = 32,
					  int stride = 1, int dilation = 1)
{
	assert(a_bit <= 7);
	assert(w_bit <= 7);

	// test_convert_uint8x16_to_uint16x8_nbits();
	// 1, 第一个改进点，将guard_bit从12比特提升到14bit，就是为了能够在最内层循环上累加更多次，从而尽可能地延迟解包计算的时间
	// 但是当我们假设最后一个数只需要6bit存储时，其实相当于58bit存储4个数，因此每个数最大可以用58/4=14bit存储
	// const int guard_bit = 14; // using the 8-bit to split the results as uchar.

	const int align_bits = 16;
	int guard_bit = 13;								 // (x-2)+4x<=64 ==> x=13 ==> 5x<=64 ==> x = 12
	int micro_iter_num = min(Ci, pow(2, guard_bit)); // 256;

	bool accelerate_with_minor_error = true;
	bool accelerate_with_major_error = false; // 由于这个方法会导致大量的计算错误，因此不使用
	if (accelerate_with_minor_error)
	{
		guard_bit = 13;
		// micro_iter_num = min(Ci, 2 * pow(2, guard_bit - 8 + 1)); // 256;
	}
	if (accelerate_with_major_error)
	{
		guard_bit = 14;
		// micro_iter_num = min(Ci, 2 * pow(2, guard_bit - 8 + 1)); // 256;
	}
#ifdef ENABLE_OPENMP
	const int regN = 4;
	const int regW = 8;
	const int regA = 2;
#else
	const int regN = 1;
	const int regW = 2;
	const int regA = 2;
#endif
	const int exbits = 1;
	const int TN = 2;

	int max_acc_3_w_mul_a = ((w_bit + a_bit) + 2);
	if (w_bit == 1 || a_bit == 1)
	{
		max_acc_3_w_mul_a -= 1;
	}

	assert(exbits + (guard_bit - max_acc_3_w_mul_a) >= 0);
	int micro_u8_iter_num = min(micro_iter_num, 2 * pow(2, exbits + (guard_bit - max_acc_3_w_mul_a))); // uint8缓存n次中间结果
	// 每个weight压缩到uint8，三个weight存储到uint32,并将weight的元素进行顺序存储
	int *packed_weights;
	posix_memalign((void **)&packed_weights, align_bits, (Co / regW + 1) * regW * K * Ci * sizeof(int));
	memset(packed_weights, 0, (Co / regW + 1) * regW * K * Ci * sizeof(int));
	for (int y = 0; y < Co; y += regW)
	{
		const int y_upper = min(regW, Co - y);
		for (int kh = 0; kh < K; kh++)
		{
			for (int i = 0; i < Ci; i++)
			{
				for (int yy = 0; yy < y_upper; yy++)
				{
					// packed_weights[y*Ci*K+i*regW*K+yy*K+kh]=0; // (y/regW)*Ci*regW+i*regW+yy
					int weight_idx = (y + yy) * Ci * K * K + i * K * K + kh * K;
					int packed_weight_idx = y * K * Ci + kh * Ci * regW + i * regW + yy; // (y/regW)*Ci*regW+i*regW+yy
					packed_weights[packed_weight_idx] = 0;
					for (int k = 0; k < K; k++)
					{
						packed_weights[packed_weight_idx] += (unsigned int)weight_ptr[weight_idx++] << (guard_bit * (K - k - 1));
					}
				}
			}
		}
	}

	// packing inputs: 在分块后可以使用常量小数组加速
	int d_Wo = Wo / K;
	int *packed_inputs;
	uint32_t size = (N / regN + 1) * regN * Ci * H * (d_Wo / regA / TN + 1) * (regA * TN) * sizeof(int);
	posix_memalign((void **)&packed_inputs, align_bits, size); // junyi: 这个地方开的内存比平常多一点，但也没关系吧。。
	memset(packed_inputs, 0, size);
	// 原始的inputs的内存顺序为[N*Ci*H*W]
	// 重新将packed_inputs排序为[H*N*Ci*d_Wo]=》
	//     这样重排仍然会导致问题，那就是当将Wo按RegA*TN进行分块后，每一次读取下一个分块的时候会跨过不连续的距离导致跳跃访问，从而导致极大的访存性能开销
	// for (int x=0;x<N;x++){
	//     for (int i=0;i<Ci;i++){
	//         for(int h=0;h<H;h++){
	//             for (int z=0;z<Wo;z+=K){
	//                 // 原始的inputs的内存顺序为[N*Ci*H*W]
	//                 // int idx = x*Ci*H*d_Wo+i*H*d_Wo+h*d_Wo+z/K;
	//                 // 重新将packed_inputs排序为[H*N*Ci*d_Wo]
	//                 int idx = h*N*Ci*d_Wo+x*Ci*d_Wo+i*d_Wo+z/K;
	//                 packed_inputs[idx] = 0;
	//                 for (int k=0;k<K;k++){
	//                     int pos = int(z*stride+k*dilation);
	//                     int inp_idx = x*Ci*H*W+i*H*W+h*W+pos;
	//                     if (pos<W)
	//                         packed_inputs[idx] += (unsigned int)inp_ptr[inp_idx]<<(guard_bit*k);
	//                 }
	//             }
	//         }
	//     }
	// }
	// 做第二次重排：
	for (int x = 0; x < N; x += regN)
	{
		const int x_upper = min(regN, N - x);
		const int x_len = N / regN;
		int x_iter = x / regN;
		for (int h = 0; h < H; h++)
		{
			for (int z = 0; z <= d_Wo; z += regA * TN)
			{ // K is the interval: 0,3,6 // <=d_Wo-TN is to
				const int z_upper = min(regA * TN, d_Wo - z);
				const int z_len = d_Wo / (regA * TN);
				int z_iter = (z / (regA * TN));

				for (int i = 0; i < Ci; i++)
				{
					for (int xx = 0; xx < x_upper; xx++)
					{
						for (int zz = 0; zz < z_upper; zz++)
						{
							// 原始的inputs的内存顺序为[N*Ci*H*W]
							// int idx = x*Ci*H*d_Wo+i*H*d_Wo+h*d_Wo+z/K;
							// 重新将packed_inputs排序为[H*N*Ci*d_Wo]
							// int idx = h*N*Ci*d_Wo+(x+xx)*Ci*d_Wo+i*d_Wo+(z+zz);
							// 设置这个是一个6维[N/regN, Ho, d_Wo/(regA*TN),Ci,x_upper,z_upper]的数组而不是4维的，否则访存永远不连续因为你的id就不连续
							int idx = x_iter * (H * z_len * Ci * regN * (regA * TN)) +
									  h * (z_len * Ci * regN * (regA * TN)) +
									  z_iter * (Ci * regN * (regA * TN)) +
									  i * regN * (regA * TN) +
									  xx * (regA * TN) +
									  zz;
							// printf("[x=%d,h=%d,z=%d,i=%d,xx=%d,zz=%d, z_upper=%d]",(x/regN),h,(z/(regA*TN)),i,xx,zz,z_upper);
							// std::cout<<"input_idx:"<<idx<<std::endl;
							int raw_z = (z + zz) * K;
							packed_inputs[idx] = 0;
							for (int k = 0; k < K; k++)
							{
								int pos = int(raw_z * stride + k * dilation);
								int inp_idx = (x + xx) * Ci * H * W + i * H * W + h * W + pos;
								if (pos < W)
									packed_inputs[idx] += (unsigned int)inp_ptr[inp_idx] << (guard_bit * k);
							}
						}
					}
				}
			}
		}
	}

	// uint64_t mask = ((1ll << guard_bit) - 1);
	// uint64_t low_mask = (mask << (guard_bit)) | (mask << (3 * guard_bit));
	// uint64_t high_mask = mask | (mask << (2 * guard_bit)) | (mask << (4 * guard_bit));
	// uint64x2_t low_maskx2 = {low_mask, low_mask};
	// uint64x2_t high_maskx2 = {high_mask, high_mask};
	uint32_t half_guard_bit = guard_bit / 2;				  // 注意，该mask会掩盖掉低guard_bit / 2位，只保留高位
	uint32_t low_half_guard_bit = guard_bit - half_guard_bit; // 注意low_mask是掩盖掉高guard_bit - half_guard_bit位，只保存低guard_bit / 2位，所以掩码得换
	uint64_t low_mask_dic = (((1ll << half_guard_bit) - 1));
	// printf("low_mask_dic:%llx\n", low_mask_dic);
	uint64_t high_mask_dic = (((1ll << low_half_guard_bit) - 1) << half_guard_bit);
	// printf("high_mask_dic:%llx\n", high_mask_dic);
	uint64_t low_mask_dic_vec = low_mask_dic | (low_mask_dic << (guard_bit)) | (low_mask_dic << (2 * guard_bit)) | (low_mask_dic << (3 * guard_bit)) | (low_mask_dic << (4 * guard_bit));
	uint64_t high_mask_dic_vec = high_mask_dic | (high_mask_dic << (guard_bit)) | (high_mask_dic << (2 * guard_bit)) | (high_mask_dic << (3 * guard_bit)) | (high_mask_dic << (4 * guard_bit));
	uint64x2_t low_mask_dic_vecx2 = {low_mask_dic_vec, low_mask_dic_vec};
	uint64x2_t high_mask_dic_vecx2 = {high_mask_dic_vec, high_mask_dic_vec};

// Test_accumlate_uint64x2x2_to_8xfloat32_nbits();
// exit(1);
// omp_set_num_threads(8); //
// #pragma omp parallel for
// 	omp_set_num_threads(4);
// #pragma omp parallel for // 展开三重循环
#ifdef ENABLE_OPENMP
	omp_set_num_threads(4);
#pragma omp parallel for
#endif
	for (int x = 0; x < N; x += regN)
	{
		const int x_upper = min(regN, N - x);
		int x_iter = x / regN;
		// #pragma omp parallel for
		for (int y = 0; y < Co; y += regW)
		{
			const int y_upper = min(regW, Co - y);
			// #pragma omp parallel for
			// int H = H/stride-(K-1)*dilation;
			// int H_upper = H - K + 1;
			int H_upper = H;
			// for (int h = 0; h < H_upper; h++)
			for (int h = -(K - 1); h < H_upper; h++)
			{ // 保持不溢出，如果需要做padding，应该在计算之前做而不是在这里做, 这里做了扩展，保证输出是最大的
				int kh_low = h < 0 ? (-h) : 0;
				int kh_upper = (H - h > K) ? K : (H - h);
				for (int kh = kh_low; kh < kh_upper; kh++)
				{
					int posh = int(h * stride + kh * dilation);
					// if (posh<0){
					//     continue;
					// }
					// if (posh>=H){
					//     break;
					// }
					// 以下这一段等价于上面的K循环
					// int upper_h = min(H,h+K);
					// for(int posh = h; posh<upper_h; posh++){
					//     int kh = posh-h;
					for (int z = 0; z <= d_Wo; z += regA * TN)
					{ // K is the interval: 0,3,6 // <=d_Wo-TN is to
						const int z_upper = min(regA * TN, d_Wo - z);
						const int z_len = d_Wo / (regA * TN);
						int z_iter = (z / (regA * TN));
						for (int i = 0; i < Ci; i += micro_iter_num)
						{
							// 取消这个缓冲来节约寄存器资源
							// float32_4 output_cache_vec[regN*regW*regA*2] = {0};
							uint64x2x2_t dual_local_accumulator[regN * regW * regA] = {0};
							// uint16_8 micro_u16_output_cache[regN*regW*regA] = {0}; // 用于缓存256次累加结果
							const int i_upper = min(micro_iter_num, Ci - i);
							for (int ii = 0; ii < i_upper; ii += micro_u8_iter_num)
							{
								const int ii_upper = min(micro_u8_iter_num, i_upper - ii);
								// kh加在这里，因为kh的计算不能在int16和int8上做累加，会溢出，但是可以在float上累加，因此放在这里

								uint64_2 micro_u8_output_cache[regN * regW * regA] = {0}; // 用于缓存2次中间累加结果 // 不能缓存4次中间结果了,因为w2a4的结果还剩2bit刚好用来累加kernel的结果了
								for (int iii = 0; iii < ii_upper; ++iii)
								{
									// 2, 第二个改进点，将input和weight的循环次序做了调换，内层循环上load weight而不是input，从而尽可能地复用input
									// 由于input需要一次load2个int32而weight仅仅需要load一个int32，可以降低内存读取？cache miss呢？
									// #pragma GCC unroll 4
									for (int xx = 0; xx < x_upper; xx++)
									{
										for (int zz = 0; zz < z_upper; zz += TN)
										{
											// 这里的内存地址不连续，多了一个常数posh*d_Wo，意味着每一轮的循环的读取，都会跳过H*d_Wo+posh*d_Wo这么长一个空位，
											// 假设H是14，d_Wo=Wo/K=4, posh=h+kh=13+2=15,那么每一次空位长度应该为(14*4+15*4)*4B=464字节，
											// 也就是说每一次读取的时候，都会跳过464字节，而L1的cache line是64字节，也就是说每次读取的时候，都会有7个cache line miss
											// L1 的cache size是32KB，也就是说每一轮iii的循环，会跳过464字节，而内存中读取的长度为regA*TN*regW*4B=32B，
											// 也就是1轮循环会跳过464+32=496字节，也就是说，每一轮的循环，会有7*496/64=56个cache line miss，
											// 每过32KB/496B=64次循环就会有一个L1的cache miss， 而我们的iii循环是2*2^(14-8+1)=256次，也就是说，每一轮的循环，会有4次L1的cache miss
											// 原始的packed_inputs内存顺序为[N*Ci*H*d_Wo]
											// int input_idx = (x+xx)*Ci*H*d_Wo+(i+ii+iii)*H*d_Wo+posh*d_Wo+z+zz;
											// 重排后packed_inputs内存顺序为[H*N*Ci*d_Wo]：从28G提升到36G
											// 重排后还是有问题：
											//     比如z=0时，每一次i循环，假设i:0->1, zz:0->2->4,真实坐标变换是（d_Wo = 28/3=10）, 0*10+0+4=4 ->1*10+0+0 = 10
											//     比如z=1时，每一次i循环，假设i:0->1, zz:0->2->4,真实坐标变换是（d_Wo = 28/3=10）, 0*10+1+4=5 ->1*10+1+0 = 11
											// 这样在内层循环中每一次都会跳过6个进行访问而不是连续的，这可能在Wo很大的时候导致极大的访存问题（跳跃访存的问题）。增加regA会部分解决该问题，但是还是无法处理大输出的问题。
											// int input_idx = posh*N*Ci*d_Wo+(x+xx)*Ci*d_Wo+(i+ii+iii)*d_Wo+(z+zz);
											int input_idx = x_iter * (H * z_len * Ci * regN * (regA * TN)) +
															posh * (z_len * Ci * regN * (regA * TN)) +
															z_iter * (Ci * regN * (regA * TN)) +
															(i + ii + iii) * regN * (regA * TN) +
															xx * (regA * TN) +
															zz;
											// int input_idx = 0;
											// std::cout<<"input_idx:"<<input_idx<<std::endl;
											uint32_2 input = load_uint32_2(packed_inputs + input_idx);
											// uint32_2 input = load_uint32_2(packed_inputs); // 这里测试如果input不存在cache miss的性能会是多少，实测可以提升3-4G左右

											for (int yy = 0; yy < y_upper; yy++)
											{
												// loading weight
												int packed_weight_idx = y * K * Ci + kh * Ci * regW + (i + ii + iii) * regW + yy; // (y/regW)*Ci*regW+i*regW+yy
												// std::cout<<"weight_idx:"<<packed_weight_idx<<std::endl;
												uint32_2 weight = load_broad_uint32_2(packed_weights[packed_weight_idx]);
												// uint32_2 weight = load_broad_uint32_2(packed_weights[0]);// 这里测试如果weight不存在cache miss的性能会是多少，实测几乎没有提升

												int cache_indx = xx * regW * regA + yy * regA + zz / TN;

												uint64_2 output = mul_low_uint32_to_uint64_(input, weight);

												// // 将10个uint8的计算结果先累加2次, 如果位宽够低还可以做更多次累加
												micro_u8_output_cache[cache_indx] = vaddq_u64(micro_u8_output_cache[cache_indx], output);
											}
										}
									}
								}
								// 将在guardbits上累加的值再次累加到2*guardbits的内存中，需要用dual向量寄存器
								for (int xx = 0; xx < x_upper; xx++)
								{
									for (int yy = 0; yy < y_upper; yy++)
									{
										for (int zz = 0; zz < z_upper; zz += TN)
										{
											int cache_indx = xx * regW * regA + yy * regA + zz / TN;
											// int output_idx = (x+xx)*Co*Ho*Wo+(y+yy)*Ho*Wo+h*Wo+(z+zz)*K;
											// 然后将10个uint12的累加结果转换为8个uint16的进行最多256次累加
											// uint16_8 output_uint16x2 = convert_uint8x16_to_uint16x8(vreinterpretq_u8_u64(micro_u8_output_cache[cache_indx]));
											// uint16_8 output_uint16x2;
											// convert_uint8x16_to_uint16x8_nbits(micro_u8_output_cache[cache_indx],output_uint16x2, guard_bit);
											// accumlate_uint16x8_to_uint16x8(output_uint16x2,micro_u16_output_cache[cache_indx]);
											// float32_4 low_output = vcvtq_f32_u32(vmovl_u16(vget_low_u16(output_uint16x2)));
											// float32_4 high_output = vcvtq_f32_u32(vmovl_u16(vget_high_u16(output_uint16x2)));
											// AddFloat4(&output_ptr[output_idx+0], low_output);
											// AddFloat4(&output_ptr[output_idx+4], high_output);
											// accumlate_uint64x2_to_8xfloat32_nbits(micro_u8_output_cache[cache_indx],&output_ptr[output_idx], guard_bit);
											// 这一步缓存在W6A6条件下提升了大约1倍的计算性能：4G->11G
											// DIC 累加低位到cache
											dual_local_accumulator[cache_indx].val[0] = vaddq_u64(dual_local_accumulator[cache_indx].val[0],
																								  vandq_u64(micro_u8_output_cache[cache_indx], low_mask_dic_vecx2));
											// 测试dual_local_accumulator[cache_indx].val[0]的值
											// uint64_t test[2];
											// vst1q_u64(test, dual_local_accumulator[cache_indx].val[0]);
											// std::cout << "test:" << test[0] << "," << test[1] << std::endl;
											dual_local_accumulator[cache_indx].val[1] = vaddq_u64(dual_local_accumulator[cache_indx].val[1],
																								  vshrq_n_u64(
																									  vandq_u64(micro_u8_output_cache[cache_indx], high_mask_dic_vecx2),
																									  half_guard_bit));
											// 测试dual_local_accumulator[cache_indx].val[1]的值
											// vst1q_u64(test, dual_local_accumulator[cache_indx].val[1]);
											// std::cout << "test:" << test[0] << "," << test[1] << std::endl;
											// exit;
										}
									}
								}
								// }
							}
							// 最后将累加结果转换为float32
							for (int xx = 0; xx < x_upper; xx++)
							{
								for (int yy = 0; yy < y_upper; yy++)
								{
									for (int zz = 0; zz < z_upper; zz += TN)
									{
										int cache_indx = xx * regW * regA + yy * regA + zz / TN;
										int output_idx = (x + xx) * Co * Ho * Wo + (y + yy) * Ho * Wo + (h + (K - 1)) * Wo + (z + zz) * K; // output_idx is decided by x,y,z. irrelated to i
										// float32_4 low_output = vcvtq_f32_u32(vmovl_u16(vget_low_u16(micro_u16_output_cache[cache_indx])));
										// float32_4 high_output = vcvtq_f32_u32(vmovl_u16(vget_high_u16(micro_u16_output_cache[cache_indx])));
										// AddFloat4(&output_ptr[output_idx+0], low_output);
										// AddFloat4(&output_ptr[output_idx+4], high_output);
										// float32_4* target_vec = output_cache_vec+(yy)*regA*2+zz/TN*2;
										// accumlate_uint16x8_to_2float32x4(micro_u16_output_cache[cache_indx],target_vec);
										accumlate_uint64x2x2_to_8xfloat32_nbits_DIC(dual_local_accumulator[cache_indx], &output_ptr[output_idx], guard_bit);
									}
								}
							}
						}
						// 累加到output上
						// for (int xx = 0; xx<x_upper; xx++){
						//     for(int yy =0;yy<y_upper;yy++){
						//         for(int zz=0;zz<z_upper;zz+=TN){
						//             int output_idx = (x+xx)*Co*Ho*Wo+(y+yy)*Ho*Wo+h*Wo+(z+zz)*K; // output_idx is decided by x,y,z. irrelated to i
						//             float32_4* target_vec = output_cache_vec+(yy)*regA*2+zz/TN*2;
						//             AddFloat4(&output_ptr[output_idx+0], target_vec[0]);
						//             AddFloat4(&output_ptr[output_idx+4], target_vec[1]);
						//         }
						//     }
						// }
					}
				}
			}
		}
	}
	std::free(packed_weights);
	std::free(packed_inputs);
}

void hipack_conv2d_khkw(const int *inp_ptr, const int *weight_ptr, float *output_ptr,
						int N, int Ci, int H, int W, int Co, int KH, int KW, int Ho, int Wo, int padding = 0,
						int a_bit = 4, int w_bit = 2, int ar_bit = 32, int wr_bit = 32,
						int stride = 1, int dilation = 1)
{

	// test_convert_uint8x16_to_uint16x8_nbits();
	// 1, 第一个改进点，将guard_bit从12比特提升到14bit，就是为了能够在最内层循环上累加更多次，从而尽可能地延迟解包计算的时间
	// 但是当我们假设最后一个数只需要6bit存储时，其实相当于58bit存储4个数，因此每个数最大可以用58/4=14bit存储
	// const int guard_bit = 14; // using the 8-bit to split the results as uchar.
	// assert(KW == 3);
	if (KW != 3)
	{
		std::cout << "Only support KW=3 while KW=" << KW << std::endl;
		return;
	}
	const int align_bits = 16;
	int guard_bit = 12;								 // (x-2)+4x<=64 ==> x=13 ==> 5x<=64 ==> x = 12
	int micro_iter_num = min(Ci, pow(2, guard_bit)); // 256;

	bool accelerate_with_minor_error = true;
	bool accelerate_with_major_error = false; // 由于这个方法会导致大量的计算错误，因此不使用
	if (accelerate_with_minor_error)
	{
		guard_bit = 13;
		// micro_iter_num = min(Ci, 2 * pow(2, guard_bit - 8 + 1)); // 256;
	}
	if (accelerate_with_major_error)
	{
		guard_bit = 14;
		// micro_iter_num = min(Ci, 2 * pow(2, guard_bit - 8 + 1)); // 256;
	}
#ifdef ENABLE_OPENMP
	const int regN = 4;
	const int regW = 8;
	const int regA = 2;
#else
	const int regN = 1;
	const int regW = 2;
	const int regA = 2;
#endif
	const int exbits = 1;
	const int TN = 2;

	int max_acc_3_w_mul_a = ((w_bit + a_bit) + 2);
	if (w_bit == 1 || a_bit == 1)
	{
		max_acc_3_w_mul_a -= 1;
	}
	int micro_u8_iter_num = min(micro_iter_num, 2 * pow(2, exbits + (guard_bit - max_acc_3_w_mul_a))); // uint8缓存n次中间结果
	// 每个weight压缩到uint8，三个weight存储到uint32,并将weight的元素进行顺序存储
	int *packed_weights;
	posix_memalign((void **)&packed_weights, align_bits, (Co / regW + 1) * regW * KH * Ci * sizeof(int));
	memset(packed_weights, 0, (Co / regW + 1) * regW * KH * Ci * sizeof(int));
	for (int y = 0; y < Co; y += regW)
	{
		const int y_upper = min(regW, Co - y);
		for (int kh = 0; kh < KH; kh++)
		{
			for (int i = 0; i < Ci; i++)
			{
				for (int yy = 0; yy < y_upper; yy++)
				{
					// packed_weights[y*Ci*K+i*regW*K+yy*K+kh]=0; // (y/regW)*Ci*regW+i*regW+yy
					int weight_idx = (y + yy) * Ci * KH * KW + i * KH * KW + kh * KW;
					int packed_weight_idx = y * KH * Ci + kh * Ci * regW + i * regW + yy; // (y/regW)*Ci*regW+i*regW+yy
					packed_weights[packed_weight_idx] = 0;
					for (int k = 0; k < KW; k++)
					{
						packed_weights[packed_weight_idx] += (unsigned int)weight_ptr[weight_idx++] << (guard_bit * (KW - k - 1));
					}
				}
			}
		}
	}

	// packing inputs: 在分块后可以使用常量小数组加速
	int d_Wo = Wo / KW;
	int *packed_inputs;
	posix_memalign((void **)&packed_inputs, align_bits, (N / regN + 1) * regN * Ci * H * (d_Wo / regA / TN + 1) * (regA * TN) * sizeof(int));
	memset(packed_inputs, 0, (N / regN + 1) * regN * Ci * H * (d_Wo / regA / TN + 1) * (regA * TN) * sizeof(int));
	// 原始的inputs的内存顺序为[N*Ci*H*W]
	// 重新将packed_inputs排序为[H*N*Ci*d_Wo]=》
	//     这样重排仍然会导致问题，那就是当将Wo按RegA*TN进行分块后，每一次读取下一个分块的时候会跨过不连续的距离导致跳跃访问，从而导致极大的访存性能开销
	// for (int x=0;x<N;x++){
	//     for (int i=0;i<Ci;i++){
	//         for(int h=0;h<H;h++){
	//             for (int z=0;z<Wo;z+=K){
	//                 // 原始的inputs的内存顺序为[N*Ci*H*W]
	//                 // int idx = x*Ci*H*d_Wo+i*H*d_Wo+h*d_Wo+z/K;
	//                 // 重新将packed_inputs排序为[H*N*Ci*d_Wo]
	//                 int idx = h*N*Ci*d_Wo+x*Ci*d_Wo+i*d_Wo+z/K;
	//                 packed_inputs[idx] = 0;
	//                 for (int k=0;k<K;k++){
	//                     int pos = int(z*stride+k*dilation);
	//                     int inp_idx = x*Ci*H*W+i*H*W+h*W+pos;
	//                     if (pos<W)
	//                         packed_inputs[idx] += (unsigned int)inp_ptr[inp_idx]<<(guard_bit*k);
	//                 }
	//             }
	//         }
	//     }
	// }
	// 做第二次重排：
	for (int x = 0; x < N; x += regN)
	{
		const int x_upper = min(regN, N - x);
		const int x_len = N / regN;
		int x_iter = x / regN;
		for (int h = 0; h < H; h++)
		{
			for (int z = 0; z <= d_Wo; z += regA * TN)
			{ // K is the interval: 0,3,6 // <=d_Wo-TN is to
				const int z_upper = min(regA * TN, d_Wo - z);
				const int z_len = d_Wo / (regA * TN);
				int z_iter = (z / (regA * TN));

				for (int i = 0; i < Ci; i++)
				{
					for (int xx = 0; xx < x_upper; xx++)
					{
						for (int zz = 0; zz < z_upper; zz++)
						{
							// 原始的inputs的内存顺序为[N*Ci*H*W]
							// int idx = x*Ci*H*d_Wo+i*H*d_Wo+h*d_Wo+z/K;
							// 重新将packed_inputs排序为[H*N*Ci*d_Wo]
							// int idx = h*N*Ci*d_Wo+(x+xx)*Ci*d_Wo+i*d_Wo+(z+zz);
							// 设置这个是一个6维[N/regN, Ho, d_Wo/(regA*TN),Ci,x_upper,z_upper]的数组而不是4维的，否则访存永远不连续因为你的id就不连续
							int idx = x_iter * (H * z_len * Ci * regN * (regA * TN)) +
									  h * (z_len * Ci * regN * (regA * TN)) +
									  z_iter * (Ci * regN * (regA * TN)) +
									  i * regN * (regA * TN) +
									  xx * (regA * TN) +
									  zz;
							// printf("[x=%d,h=%d,z=%d,i=%d,xx=%d,zz=%d, z_upper=%d]",(x/regN),h,(z/(regA*TN)),i,xx,zz,z_upper);
							// std::cout<<"input_idx:"<<idx<<std::endl;
							int raw_z = (z + zz) * KW;
							packed_inputs[idx] = 0;
							for (int k = 0; k < KW; k++)
							{
								int pos = int(raw_z * stride + k * dilation);
								int inp_idx = (x + xx) * Ci * H * W + i * H * W + h * W + pos;
								if (pos < W)
									packed_inputs[idx] += (unsigned int)inp_ptr[inp_idx] << (guard_bit * k);
							}
						}
					}
				}
			}
		}
	}

	// uint64_t mask = ((1ll << guard_bit) - 1);
	// uint64_t low_mask = (mask << (guard_bit)) | (mask << (3 * guard_bit));
	// uint64_t high_mask = mask | (mask << (2 * guard_bit)) | (mask << (4 * guard_bit));
	// uint64x2_t low_maskx2 = {low_mask, low_mask};
	// uint64x2_t high_maskx2 = {high_mask, high_mask};
	uint32_t half_guard_bit = guard_bit / 2;				  // 注意，该mask会掩盖掉低guard_bit / 2位，只保留高位
	uint32_t low_half_guard_bit = guard_bit - half_guard_bit; // 注意low_mask是掩盖掉高guard_bit - half_guard_bit位，只保存低guard_bit / 2位，所以掩码得换
	uint64_t low_mask_dic = (((1ll << half_guard_bit) - 1));
	// printf("low_mask_dic:%llx\n", low_mask_dic);
	uint64_t high_mask_dic = (((1ll << low_half_guard_bit) - 1) << half_guard_bit);
	// printf("high_mask_dic:%llx\n", high_mask_dic);
	uint64_t low_mask_dic_vec = low_mask_dic | (low_mask_dic << (guard_bit)) | (low_mask_dic << (2 * guard_bit)) | (low_mask_dic << (3 * guard_bit)) | (low_mask_dic << (4 * guard_bit));
	uint64_t high_mask_dic_vec = high_mask_dic | (high_mask_dic << (guard_bit)) | (high_mask_dic << (2 * guard_bit)) | (high_mask_dic << (3 * guard_bit)) | (high_mask_dic << (4 * guard_bit));
	uint64x2_t low_mask_dic_vecx2 = {low_mask_dic_vec, low_mask_dic_vec};
	uint64x2_t high_mask_dic_vecx2 = {high_mask_dic_vec, high_mask_dic_vec};

// Test_accumlate_uint64x2x2_to_8xfloat32_nbits();
// exit(1);
// omp_set_num_threads(8); //
// #pragma omp parallel for
// 	omp_set_num_threads(4);
// #pragma omp parallel for // 展开三重循环
#ifdef ENABLE_OPENMP
	omp_set_num_threads(4);
#pragma omp parallel for
#endif
	for (int x = 0; x < N; x += regN)
	{
		const int x_upper = min(regN, N - x);
		int x_iter = x / regN;
		// #pragma omp parallel for
		for (int y = 0; y < Co; y += regW)
		{
			const int y_upper = min(regW, Co - y);
			// #pragma omp parallel for
			// int H = H/stride-(K-1)*dilation;
			// int H_upper = H - K + 1;
			int H_upper = H;
			// for (int h = 0; h < H_upper; h++)
			for (int h = -(KH - 1); h < H_upper; h++)
			{ // 保持不溢出，如果需要做padding，应该在计算之前做而不是在这里做, 这里做了扩展，保证输出是最大的
				int kh_low = h < 0 ? (-h) : 0;
				int kh_upper = (H - h > KH) ? KH : (H - h);
				for (int kh = kh_low; kh < kh_upper; kh++)
				{
					int posh = int(h * stride + kh * dilation);
					// if (posh<0){
					//     continue;
					// }
					// if (posh>=H){
					//     break;
					// }
					// 以下这一段等价于上面的K循环
					// int upper_h = min(H,h+K);
					// for(int posh = h; posh<upper_h; posh++){
					//     int kh = posh-h;
					for (int z = 0; z <= d_Wo; z += regA * TN)
					{ // K is the interval: 0,3,6 // <=d_Wo-TN is to
						const int z_upper = min(regA * TN, d_Wo - z);
						const int z_len = d_Wo / (regA * TN);
						int z_iter = (z / (regA * TN));
						for (int i = 0; i < Ci; i += micro_iter_num)
						{
							// 取消这个缓冲来节约寄存器资源
							// float32_4 output_cache_vec[regN*regW*regA*2] = {0};
							uint64x2x2_t dual_local_accumulator[regN * regW * regA] = {0};
							// uint16_8 micro_u16_output_cache[regN*regW*regA] = {0}; // 用于缓存256次累加结果
							const int i_upper = min(micro_iter_num, Ci - i);
							for (int ii = 0; ii < i_upper; ii += micro_u8_iter_num)
							{
								const int ii_upper = min(micro_u8_iter_num, i_upper - ii);
								// kh加在这里，因为kh的计算不能在int16和int8上做累加，会溢出，但是可以在float上累加，因此放在这里

								uint64_2 micro_u8_output_cache[regN * regW * regA] = {0}; // 用于缓存2次中间累加结果 // 不能缓存4次中间结果了,因为w2a4的结果还剩2bit刚好用来累加kernel的结果了
								for (int iii = 0; iii < ii_upper; ++iii)
								{
									// 2, 第二个改进点，将input和weight的循环次序做了调换，内层循环上load weight而不是input，从而尽可能地复用input
									// 由于input需要一次load2个int32而weight仅仅需要load一个int32，可以降低内存读取？cache miss呢？
									// #pragma GCC unroll 4
									for (int xx = 0; xx < x_upper; xx++)
									{
										for (int zz = 0; zz < z_upper; zz += TN)
										{
											// 这里的内存地址不连续，多了一个常数posh*d_Wo，意味着每一轮的循环的读取，都会跳过H*d_Wo+posh*d_Wo这么长一个空位，
											// 假设H是14，d_Wo=Wo/K=4, posh=h+kh=13+2=15,那么每一次空位长度应该为(14*4+15*4)*4B=464字节，
											// 也就是说每一次读取的时候，都会跳过464字节，而L1的cache line是64字节，也就是说每次读取的时候，都会有7个cache line miss
											// L1 的cache size是32KB，也就是说每一轮iii的循环，会跳过464字节，而内存中读取的长度为regA*TN*regW*4B=32B，
											// 也就是1轮循环会跳过464+32=496字节，也就是说，每一轮的循环，会有7*496/64=56个cache line miss，
											// 每过32KB/496B=64次循环就会有一个L1的cache miss， 而我们的iii循环是2*2^(14-8+1)=256次，也就是说，每一轮的循环，会有4次L1的cache miss
											// 原始的packed_inputs内存顺序为[N*Ci*H*d_Wo]
											// int input_idx = (x+xx)*Ci*H*d_Wo+(i+ii+iii)*H*d_Wo+posh*d_Wo+z+zz;
											// 重排后packed_inputs内存顺序为[H*N*Ci*d_Wo]：从28G提升到36G
											// 重排后还是有问题：
											//     比如z=0时，每一次i循环，假设i:0->1, zz:0->2->4,真实坐标变换是（d_Wo = 28/3=10）, 0*10+0+4=4 ->1*10+0+0 = 10
											//     比如z=1时，每一次i循环，假设i:0->1, zz:0->2->4,真实坐标变换是（d_Wo = 28/3=10）, 0*10+1+4=5 ->1*10+1+0 = 11
											// 这样在内层循环中每一次都会跳过6个进行访问而不是连续的，这可能在Wo很大的时候导致极大的访存问题（跳跃访存的问题）。增加regA会部分解决该问题，但是还是无法处理大输出的问题。
											// int input_idx = posh*N*Ci*d_Wo+(x+xx)*Ci*d_Wo+(i+ii+iii)*d_Wo+(z+zz);
											int input_idx = x_iter * (H * z_len * Ci * regN * (regA * TN)) +
															posh * (z_len * Ci * regN * (regA * TN)) +
															z_iter * (Ci * regN * (regA * TN)) +
															(i + ii + iii) * regN * (regA * TN) +
															xx * (regA * TN) +
															zz;
											// int input_idx = 0;
											// std::cout<<"input_idx:"<<input_idx<<std::endl;
											uint32_2 input = load_uint32_2(packed_inputs + input_idx);
											// uint32_2 input = load_uint32_2(packed_inputs); // 这里测试如果input不存在cache miss的性能会是多少，实测可以提升3-4G左右

											for (int yy = 0; yy < y_upper; yy++)
											{
												// loading weight
												int packed_weight_idx = y * KH * Ci + kh * Ci * regW + (i + ii + iii) * regW + yy; // (y/regW)*Ci*regW+i*regW+yy
												// std::cout<<"weight_idx:"<<packed_weight_idx<<std::endl;
												uint32_2 weight = load_broad_uint32_2(packed_weights[packed_weight_idx]);
												// uint32_2 weight = load_broad_uint32_2(packed_weights[0]);// 这里测试如果weight不存在cache miss的性能会是多少，实测几乎没有提升

												int cache_indx = xx * regW * regA + yy * regA + zz / TN;

												uint64_2 output = mul_low_uint32_to_uint64_(input, weight);

												// // 将10个uint8的计算结果先累加2次, 如果位宽够低还可以做更多次累加
												micro_u8_output_cache[cache_indx] = vaddq_u64(micro_u8_output_cache[cache_indx], output);
											}
										}
									}
								}
								// 将在guardbits上累加的值再次累加到2*guardbits的内存中，需要用dual向量寄存器
								for (int xx = 0; xx < x_upper; xx++)
								{
									for (int yy = 0; yy < y_upper; yy++)
									{
										for (int zz = 0; zz < z_upper; zz += TN)
										{
											int cache_indx = xx * regW * regA + yy * regA + zz / TN;
											// int output_idx = (x+xx)*Co*Ho*Wo+(y+yy)*Ho*Wo+h*Wo+(z+zz)*K;
											// 然后将10个uint12的累加结果转换为8个uint16的进行最多256次累加
											// uint16_8 output_uint16x2 = convert_uint8x16_to_uint16x8(vreinterpretq_u8_u64(micro_u8_output_cache[cache_indx]));
											// uint16_8 output_uint16x2;
											// convert_uint8x16_to_uint16x8_nbits(micro_u8_output_cache[cache_indx],output_uint16x2, guard_bit);
											// accumlate_uint16x8_to_uint16x8(output_uint16x2,micro_u16_output_cache[cache_indx]);
											// float32_4 low_output = vcvtq_f32_u32(vmovl_u16(vget_low_u16(output_uint16x2)));
											// float32_4 high_output = vcvtq_f32_u32(vmovl_u16(vget_high_u16(output_uint16x2)));
											// AddFloat4(&output_ptr[output_idx+0], low_output);
											// AddFloat4(&output_ptr[output_idx+4], high_output);
											// accumlate_uint64x2_to_8xfloat32_nbits(micro_u8_output_cache[cache_indx],&output_ptr[output_idx], guard_bit);
											// 这一步缓存在W6A6条件下提升了大约1倍的计算性能：4G->11G
											// DIC 累加低位到cache
											dual_local_accumulator[cache_indx].val[0] = vaddq_u64(dual_local_accumulator[cache_indx].val[0],
																								  vandq_u64(micro_u8_output_cache[cache_indx], low_mask_dic_vecx2));
											// 测试dual_local_accumulator[cache_indx].val[0]的值
											// uint64_t test[2];
											// vst1q_u64(test, dual_local_accumulator[cache_indx].val[0]);
											// std::cout << "test:" << test[0] << "," << test[1] << std::endl;
											dual_local_accumulator[cache_indx].val[1] = vaddq_u64(dual_local_accumulator[cache_indx].val[1],
																								  vshrq_n_u64(
																									  vandq_u64(micro_u8_output_cache[cache_indx], high_mask_dic_vecx2),
																									  half_guard_bit));
											// 测试dual_local_accumulator[cache_indx].val[1]的值
											// vst1q_u64(test, dual_local_accumulator[cache_indx].val[1]);
											// std::cout << "test:" << test[0] << "," << test[1] << std::endl;
											// exit;
										}
									}
								}
								// }
							}
							// 最后将累加结果转换为float32
							for (int xx = 0; xx < x_upper; xx++)
							{
								for (int yy = 0; yy < y_upper; yy++)
								{
									for (int zz = 0; zz < z_upper; zz += TN)
									{
										int cache_indx = xx * regW * regA + yy * regA + zz / TN;
										int output_idx = (x + xx) * Co * Ho * Wo + (y + yy) * Ho * Wo + (h + (KH - 1)) * Wo + (z + zz) * KW; // output_idx is decided by x,y,z. irrelated to i
										// float32_4 low_output = vcvtq_f32_u32(vmovl_u16(vget_low_u16(micro_u16_output_cache[cache_indx])));
										// float32_4 high_output = vcvtq_f32_u32(vmovl_u16(vget_high_u16(micro_u16_output_cache[cache_indx])));
										// AddFloat4(&output_ptr[output_idx+0], low_output);
										// AddFloat4(&output_ptr[output_idx+4], high_output);
										// float32_4* target_vec = output_cache_vec+(yy)*regA*2+zz/TN*2;
										// accumlate_uint16x8_to_2float32x4(micro_u16_output_cache[cache_indx],target_vec);
										accumlate_uint64x2x2_to_8xfloat32_nbits_DIC(dual_local_accumulator[cache_indx], &output_ptr[output_idx], guard_bit);
									}
								}
							}
						}
						// 累加到output上
						// for (int xx = 0; xx<x_upper; xx++){
						//     for(int yy =0;yy<y_upper;yy++){
						//         for(int zz=0;zz<z_upper;zz+=TN){
						//             int output_idx = (x+xx)*Co*Ho*Wo+(y+yy)*Ho*Wo+h*Wo+(z+zz)*K; // output_idx is decided by x,y,z. irrelated to i
						//             float32_4* target_vec = output_cache_vec+(yy)*regA*2+zz/TN*2;
						//             AddFloat4(&output_ptr[output_idx+0], target_vec[0]);
						//             AddFloat4(&output_ptr[output_idx+4], target_vec[1]);
						//         }
						//     }
						// }
					}
				}
			}
		}
	}
	std::free(packed_weights);
	std::free(packed_inputs);
}
