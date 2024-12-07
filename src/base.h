#pragma once
#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <arm_neon.h>
#include <cassert>
#include <pthread.h>
#include <cstring> // 包含memset的头文件
#include <pthreadpool.h>
#include <chrono>
#include <cstdlib>
// random number generator
#include <ctime>
// other
#include <iterator>
#include <tuple>
#include <vector>
#include <algorithm>
#include <numeric> // Add this line to include the <numeric> header
#include <random>
#ifdef ENABLE_OPENMP
#include <omp.h>
#endif
using namespace std;

#define explicit_grid_loop(x, X) \
	for (x = 0; x < X; x++)

#define explicit_grid_loop2(x, y, X, Y) \
	for (x = 0; x < X; x++)             \
		for (y = 0; y < Y; y++)

#define explicit_grid_loop3(x, y, z, X, Y, Z) \
	for (x = 0; x < X; x++)                   \
		for (y = 0; y < Y; y++)               \
			for (z = 0; z < Z; z++)
#define explicit_grid_loop4(x, y, z, w, X, Y, Z, W) \
	for (x = 0; x < X; x++)                         \
		for (y = 0; y < Y; y++)                     \
			for (z = 0; z < Z; z++)                 \
				for (w = 0; w < W; w++)
#define explicit_grid_loop5(x, y, z, w, v, X, Y, Z, W, V) \
	for (x = 0; x < X; x++)                               \
		for (y = 0; y < Y; y++)                           \
			for (z = 0; z < Z; z++)                       \
				for (w = 0; w < W; w++)                   \
					for (v = 0; v < V; v++)
#define explicit_grid_loop6(x, y, z, w, v, u, X, Y, Z, W, V, U) \
	for (x = 0; x < X; x++)                                     \
		for (y = 0; y < Y; y++)                                 \
			for (z = 0; z < Z; z++)                             \
				for (w = 0; w < W; w++)                         \
					for (v = 0; v < V; v++)                     \
						for (u = 0; u < U; u++)

#define anonymous_grid_loop(x, X) \
	for (int x = 0; x < X; x++)

#define grid_loop(x, X) anonymous_grid_loop(x, X)

#define anonymous_grid_loop2(x, y, X, Y) \
	for (int x = 0; x < X; x++)          \
		for (int y = 0; y < Y; y++)
#define grid_loop2(x, y, X, Y) anonymous_grid_loop2(x, y, X, Y)

#define anonymous_grid_loop3(x, y, z, X, Y, Z) \
	for (int x = 0; x < X; x++)                \
		for (int y = 0; y < Y; y++)            \
			for (int z = 0; z < Z; z++)
#define grid_loop3(x, y, z, X, Y, Z) anonymous_grid_loop3(x, y, z, X, Y, Z)

#define anonymous_grid_loop4(x, y, z, w, X, Y, Z, W) \
	for (int x = 0; x < X; x++)                      \
		for (int y = 0; y < Y; y++)                  \
			for (int z = 0; z < Z; z++)              \
				for (int w = 0; w < W; w++)
#define grid_loop4(x, y, z, w, X, Y, Z, W) anonymous_grid_loop4(x, y, z, w, X, Y, Z, W)

#define anonymous_grid_loop5(x, y, z, w, v, X, Y, Z, W, V) \
	for (int x = 0; x < X; x++)                            \
		for (int y = 0; y < Y; y++)                        \
			for (int z = 0; z < Z; z++)                    \
				for (int w = 0; w < W; w++)                \
					for (int v = 0; v < V; v++)
#define grid_loop5(x, y, z, w, v, X, Y, Z, W, V) anonymous_grid_loop5(x, y, z, w, v, X, Y, Z, W, V)

#define anonymous_grid_loop6(x, y, z, w, v, u, X, Y, Z, W, V, U) \
	for (int x = 0; x < X; x++)                                  \
		for (int y = 0; y < Y; y++)                              \
			for (int z = 0; z < Z; z++)                          \
				for (int w = 0; w < W; w++)                      \
					for (int v = 0; v < V; v++)                  \
						for (int u = 0; u < U; u++)
#define grid_loop6(x, y, z, w, v, u, X, Y, Z, W, V, U) anonymous_grid_loop6(x, y, z, w, v, u, X, Y, Z, W, V, U)

typedef unsigned long long uint64;
typedef unsigned int uint32;
typedef unsigned char uint8;

// simd hiconv
typedef uint16_t uint16;
typedef uint64x2_t uint64_2;
typedef uint32x4_t uint32_4;
typedef uint32x2_t uint32_2;
typedef uint16x8_t uint16_8;
typedef float32x4_t float32_4;
#define load_uint32_4(mem_addr) vld1q_u32((const uint32_t *)(mem_addr))
#define load_uint32_2(mem_addr) vld1_u32((const uint32_t *)(mem_addr))
#define load_uint16_8(mem_addr) vld1q_u16((const uint16_t *)(mem_addr))
#define load_uint16_4(mem_addr) vld1_u16((const uint16_t *)(mem_addr))
#define load_uint64_2(mem_addr) vld1q_u64((const uint64_t *)(mem_addr))
#define load_broad_uint64_2(val) vdupq_n_u64((uint64)(val))
#define load_broad_uint32_4(val) vdupq_n_u32((uint32)(val))
#define load_broad_uint32_2(val) vdup_n_u32((uint32)(val))
#define load_broad_uint16_4(val) vdup_n_u16((uint16_t)(val))
#define load_broad_uint16_8(val) vdupq_n_u16((uint16_t)(val))
#define accumlate_uint64x2_to_uint64x2(input, target) (target = vaddq_u64(input, target))
#define accumlate_uint16x8_to_uint16x8(input, target) (target = vaddq_u16(input, target))

// input: uint16x8 target: 2xfloat32x4
#define accumlate_uint16x8_to_2float32x4(input, target)                         \
	{                                                                           \
		float32_4 low_output = vcvtq_f32_u32(vmovl_u16(vget_low_u16(input)));   \
		float32_4 high_output = vcvtq_f32_u32(vmovl_u16(vget_high_u16(input))); \
		target[0] = vaddq_f32(target[0], low_output);                           \
		target[1] = vaddq_f32(target[1], high_output);                          \
	}

#ifndef AddFloat4
#define AddFloat4(ptr, vec)                                \
	{                                                      \
		float32x4_t vec1 = vaddq_f32(vec, vld1q_f32(ptr)); \
		vst1q_f32(ptr, vec1);                              \
	}
#define StoreFloat4(ptr, vec) vst1q_f32(ptr, vec);
#endif

#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

// neon没有int64的乘法指令
// #define mul_uint64_2_to_uint64_2(uint64_2_val1,uint64_2_val2) vmulq_u64((uint64_2_val1),(uint64_2_val2))
// #define mul_low_uint32_to_uint64(uint64_2_val1, uint64_2_val2,target_out) \
// ({ \
//     uint32x2_t lo1 = vget_low_u32(vreinterpretq_u32_u64(uint64_2_val1)); \
//     uint32x2_t lo2 = vget_low_u32(vreinterpretq_u32_u64(uint64_2_val2)); \
//     target_out = vmull_u32(lo1, lo2); \
// ({ \

#define mul_low_uint32_to_uint64(uint32_2_val1, uint32_2_val2, target_out) (target_out = vmull_u32(uint32_2_val1, uint32_2_val2))
#define mul_low_uint32_to_uint64_(uint32_2_val1, uint32_2_val2) (vmull_u32(uint32_2_val1, uint32_2_val2))
#define mul_low_uint16x4_to_uint32x4_(uint16_4_val1, uint16_4_val2) (vmull_u16(uint16_4_val1, uint16_4_val2))

#define failed_words "Test \033[31m" << "failed!" << "\033[0m"
#define passed_words "Test \033[32m" << "pass!" << "\033[0m"

// 确保在程序的开始处调用此函数一次
void initRandomSeed()
{
	srand(time(0)); // 使用当前时间作为种子
}

inline float getRand(int min, int max)
{
	return (rand() % (max - min + 1)) + min;
}
inline double getTime(auto start, auto end)
{
	// if (end==0) end = std::chrono::system_clock::now(); //std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	double time = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
	return time;
}
inline double getTime(auto start)
{
	auto end = std::chrono::system_clock::now(); // std::chrono::high_resolution_clock::now();
	return getTime(start, end);
}

// 包含两个uint64，每个uint64包含5x12bit的无符号整型，
inline void accumlate_uint32x4_to_14xfloat32_nbits(uint32x4_t &input, float *output_ptr, const int guard_bit = 12, bool debug = false)
{
	if (debug)
		return;
	uint32_t vec_values[4];
	vst1q_u32(vec_values, input);
	uint32_t mask = ((1 << guard_bit) - 1);
	// 这几组32bit的数前一个32的高21-31位累加到后一个32位的尾巴上
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			output_ptr[i * 3 + j] += (float)((vec_values[i] >> (j * guard_bit)) & mask);
		}
	}
}

// 包含两个uint64，每个uint64包含5x12bit的无符号整型，
inline void accumlate_uint64x2_to_8xfloat32_nbits(uint64x2_t &input, float *output_ptr, const int guard_bit = 12, bool debug = false)
{
	if (debug)
		return;
	uint64_t vec_values[2];
	vst1q_u64(vec_values, input);
	uint32_t mask = ((1 << guard_bit) - 1);
	// 这几组32bit的数前一个32的高21-31位累加到后一个32位的尾巴上
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			output_ptr[i * 3 + j] += (float)((vec_values[i] >> (j * guard_bit)) & mask);
		}
	}
}

inline void accumlate_uint32x4x2_to_14xfloat32_nbits(uint32x4x2_t &input, float *output_ptr, const int guard_bit = 7, bool debug = false)
{
	if (debug)
		return;
	// input1构成是4x3xresult(4个int32，每个int32包含3个results),每个位宽2*guard_bit, offset=0,以input1[0/1][0](128->64->2*guard_bit),input1[0/1][1],input1[0/1][2]
	// input2构成是4x2xresult,每个位宽2*guard_bit, offset=guard_bit,以input2[0/1][0],input2[0/1][1]
	uint32_t vec_values[2][4];
	vst2q_u32(vec_values[0], input);
	// vst1q_u64(vec_values[1],input.val[1]);
	uint32_t mask = ((1 << (2 * guard_bit)) - 1);
	uint32_t offset = guard_bit;
	// 这几组32bit的数前一个32的高21-31位累加到后一个32位的尾巴上
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			int j_index = j % 2;
			int value = (vec_values[j_index][i] >> (j * guard_bit)) & mask;
			// printf("value[%d,%d]=%d\n",i,j,value);
			output_ptr[i * 3 + j] += (float)(value);
		}
	}
}

void Test_accumlate_uint32x4x2_to_14xfloat32_nbits()
{
	// build test data
	uint32x4x2_t input;
	uint32_t input_val[2][4] = {
		{0, 0, 0, 0},
		{0, 0, 0, 0}};
	int guard_bit = 7;
	input_val[0][0] = 32 + (64 << (2 * guard_bit)) + (128 << (4 * guard_bit));
	input_val[1][0] = (48 << (guard_bit)) + (96 << (3 * guard_bit));
	input_val[0][1] = 16 + (64 << (2 * guard_bit)) + (128ll << (4 * guard_bit));
	input_val[1][1] = (48 << (guard_bit)) + (96ll << (3 * guard_bit));
	input = vld2q_u32(reinterpret_cast<uint32_t *>(input_val[0]));
	// input.val[1] = vld2q_u64(reinterpret_cast<uint64_t*>(input_val[1]));
	float output_ref[8];
	float output[8];
	for (int i = 0; i < 8; i++)
		output[i] = 0;
	// Test
	accumlate_uint32x4x2_to_14xfloat32_nbits(input, output, guard_bit, false);
}

// 包含两个uint64x2_t，每个uint64包含5x12bit的无符号整型，
/**
 * @brief 将 uint64x2x2_t 累加到 8 个 float32 中（带位数）
 *
 * 将给定的 uint64x2x2_t 类型的输入进行累加，并将结果存储到输出指针指向的 float 数组中。
 * 输入由两个 uint64x2_t 组成，每个 uint64x2_t 包含两个uint64_t，每个uint64_t包含三个结果，每个结果的位宽为 2 * guard_bit。
 * 输出为 8 个 float 类型的数值，用于存储累加结果。
 *
 * @param input uint64x2x2_t 类型的引用，表示输入数据
 * @param output_ptr float 类型的指针，表示输出数据的存储位置
 * @param guard_bit 整数类型，表示每个结果的保护位数，默认为 12
 * @param debug 布尔类型，表示是否开启调试模式，默认为 false
 */
inline void accumlate_uint64x2x2_to_8xfloat32_nbits(uint64x2x2_t &input, float *output_ptr, const int guard_bit = 12, bool debug = false)
{
	if (debug)
		return;
	// input1构成是2x3xresult(两个int64，每个int64包含3个results),每个位宽2*guard_bit, offset=0,以input1[0/1][0](128->64->2*guard_bit),input1[0/1][1],input1[0/1][2]
	// input2构成是2x2xresult,每个位宽2*guard_bit, offset=guard_bit,以input2[0/1][0],input2[0/1][1]
	uint64_t vec_values[2][2];
	vst1q_u64(vec_values[0], input.val[0]);
	vst1q_u64(vec_values[1], input.val[1]);
	// vst1q_u64(vec_values[1],input.val[1]);
	uint32_t mask = ((1 << (2 * guard_bit)) - 1);
	uint32_t offset = guard_bit;
	// 这几组32bit的数前一个32的高21-31位累加到后一个32位的尾巴上
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			int j_index = j % 2;
			int value = (vec_values[j_index][i] >> (j * guard_bit)) & mask;
			// printf("value[%d,%d]=%d\n",i,j,value);
			output_ptr[i * 3 + j] += (float)(value);
		}
	}
}

void Test_accumlate_uint64x2x2_to_8xfloat32_nbits()
{
	// build test data
	uint64x2x2_t input;
	uint64_t input_val[2][2] = {
		{0, 0},
		{0, 0}};
	int guard_bit = 12;
	input_val[0][0] = 32ll + (64 << (2 * guard_bit)) + (128ll << (4 * guard_bit));
	input_val[1][0] = (48 << (guard_bit)) + (96ll << (3 * guard_bit));
	input_val[0][1] = 16 + (64 << (2 * guard_bit)) + (128ll << (4 * guard_bit));
	input_val[1][1] = (48 << (guard_bit)) + (96ll << (3 * guard_bit));
	input = vld2q_u64(reinterpret_cast<uint64_t *>(input_val[0]));
	// input.val[1] = vld2q_u64(reinterpret_cast<uint64_t*>(input_val[1]));
	float output_ref[8];
	float output[8];
	for (int i = 0; i < 8; i++)
		output[i] = 0;
	// Test
	accumlate_uint64x2x2_to_8xfloat32_nbits(input, output, guard_bit, false);
}

/**
 * @brief 这个函数实现DIC，将两个uint64x2x2_t中的低位和高位分别累加到8个float32中（带位数）
 *
 * 将给定的 uint64x2x2_t 类型的输入进行累加，并将结果存储到输出指针指向的 float 数组中。
 * 输入由两个 uint64x2_t 组成，每个 uint64x2_t 包含两个uint64_t，每个uint64_t包含五个结果，每个结果的位宽为 guard_bit。
 * 在两个uint64x2x2_t的每一个uint64x2_t数据中，总共存储了10个位宽为guard_bit的整型值，其中第一个uint64x2_t存储的是10个完整整型值的低位，第二个存储的是高位，比低位高出guard_bit//2这么多位。
 * 输出为 8 个 float 类型的数值，用于存储累加结果。
 * 在本函数中，我们需要将10个低位和10个高位分别取出来，然后将第4个和第6个累加为一个，第5和第7个累加为一个，然后总共累加到8个float32中，具体算法如下：
 * f[0] = u64x2[0][0]+(u64x2[1][0]<<(guard_bit//2))
 * f[1] = u64x2[0][1]+(u64x2[1][1]<<(guard_bit//2))
 * f[2] = u64x2[0][2]+(u64x2[1][2]<<(guard_bit//2))
 * f[3] = u64x2[0][3]+(u64x2[1][3]<<(guard_bit//2)) + u64x2[0][5]+(u64x2[1][5]<<(guard_bit//2)) // 将第4个和第6个累加为一个
 * f[4] = u64x2[0][4]+(u64x2[1][4]<<(guard_bit//2)) + u64x2[0][6]+(u64x2[1][6]<<(guard_bit//2)) // 将第5个和第7个累加为一个
 * f[5] = u64x2[0][7]+(u64x2[1][7]<<(guard_bit//2))
 * f[6] = u64x2[0][8]+(u64x2[1][8]<<(guard_bit//2))
 * f[7] = u64x2[0][9]+(u64x2[1][9]<<(guard_bit//2))
 * 以下是输入：
 * @param input uint64x2x2_t 类型的引用，表示输入数据
 * @param output_ptr float 类型的指针，表示输出数据的存储位置
 * @param guard_bit 整数类型，表示每个结果的保护位数，默认为 12
 * @param debug 布尔类型，表示是否开启调试模式，默认为 false
 */
inline void accumlate_uint64x2x2_to_8xfloat32_nbits_DIC(uint64x2x2_t &input, float *output_ptr, const int guard_bit = 12, bool debug = false)
{
	if (debug)
		return;
	// input1构成是2x3xresult(两个int64，每个int64包含3个results),每个位宽2*guard_bit, offset=0,以input1[0/1][0](128->64->2*guard_bit),input1[0/1][1],input1[0/1][2]
	// input2构成是2x2xresult,每个位宽2*guard_bit, offset=guard_bit,以input2[0/1][0],input2[0/1][1]
	uint64_t vec_values[2][2];
	vst1q_u64(vec_values[0], input.val[0]);
	vst1q_u64(vec_values[1], input.val[1]);
	// vst1q_u64(vec_values[1],input.val[1]);
	uint32_t mask = ((1 << (guard_bit)) - 1);
	uint32_t high_offset = guard_bit / 2;
	// uint32_t high_mask = ((1 << (guard_bit)) - 1) << (guard_bit / 2); // 0x0f0：高位存的值比低位存的值高出guard_bit//2这么多位
	uint32_t global_offset = guard_bit;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			// low uint64 = vec_values[0][i], high uint64 = vec_values[1][i]
			int low_value = (vec_values[0][i] >> (j * global_offset)) & mask;
			int high_value = (vec_values[1][i] >> (j * global_offset)) & mask;
			int value = low_value + (high_value << high_offset);
			// printf("low value[0,%d,%d]=%d", i, j, low_value);
			// printf(", high value[1,%d,%d]=%d", i, j, high_value);
			// printf(", Added value[%d,%d]=%d\n", i, j, value);
			output_ptr[i * 3 + j] += (float)(value);
		}
	}
}

bool Test_accumlate_uint64x2x2_to_8xfloat32_nbits_DIC()
{
	// build test data
	uint64x2x2_t input;
	uint64_t input_val[2][2] = {
		{0, 0},
		{0, 0}};
	int guard_bit = 12;

	// The first uint64x2_t
	input_val[0][0] = 1ll + (2ll << (guard_bit)) + (3ll << (2 * guard_bit)) + (4ll << (3 * guard_bit)) + +(5ll << (4 * guard_bit));
	input_val[1][0] = 6ll + (7ll << (guard_bit)) + (8ll << (2 * guard_bit)) + (9ll << (3 * guard_bit)) + +(10ll << (4 * guard_bit));
	// The second uint64x2_t
	input_val[0][1] = 1ll + (2ll << (guard_bit)) + (3ll << (2 * guard_bit)) + (4ll << (3 * guard_bit)) + +(5ll << (4 * guard_bit));
	input_val[1][1] = 6ll + (7ll << (guard_bit)) + (8ll << (2 * guard_bit)) + (9ll << (3 * guard_bit)) + +(10ll << (4 * guard_bit));
	// 请注意vld2q_u64函数是交错加载的，即第1和3个会加载到第一个uint64x2_t中，第2和4个会加载到第二个uint64x2_t中
	input = vld2q_u64(reinterpret_cast<uint64_t *>(input_val[0]));
	// input.val[1] = vld2q_u64(reinterpret_cast<uint64_t*>(input_val[1]));
	float output_ref[8];
	/*
	 * f[0] = u64x2[0][0]+(u64x2[1][0]<<(guard_bit//2))
	 * f[1] = u64x2[0][1]+(u64x2[1][1]<<(guard_bit//2))
	 * f[2] = u64x2[0][2]+(u64x2[1][2]<<(guard_bit//2))
	 * f[3] = u64x2[0][3]+(u64x2[1][3]<<(guard_bit//2)) + u64x2[0][5]+(u64x2[1][5]<<(guard_bit//2)) // 将第4个和第6个累加为一个
	 * f[4] = u64x2[0][4]+(u64x2[1][4]<<(guard_bit//2)) + u64x2[0][6]+(u64x2[1][6]<<(guard_bit//2)) // 将第5个和第7个累加为一个
	 * f[5] = u64x2[0][7]+(u64x2[1][7]<<(guard_bit//2))
	 * f[6] = u64x2[0][8]+(u64x2[1][8]<<(guard_bit//2))
	 * f[7] = u64x2[0][9]+(u64x2[1][9]<<(guard_bit//2))
	 */
	output_ref[0] = 1 + (1 << (guard_bit / 2));
	output_ref[1] = 2 + (2 << (guard_bit / 2));
	output_ref[2] = 3 + (3 << (guard_bit / 2));
	output_ref[3] = 4 + (4 << (guard_bit / 2)) + 6 + (6 << (guard_bit / 2));
	output_ref[4] = 5 + (5 << (guard_bit / 2)) + 7 + (7 << (guard_bit / 2));
	output_ref[5] = 8 + (8 << (guard_bit / 2));
	output_ref[6] = 9 + (9 << (guard_bit / 2));
	output_ref[7] = 10 + (10 << (guard_bit / 2));

	float output[8];
	for (int i = 0; i < 8; i++)
	{
		output[i] = 0;
	}
	// Test
	accumlate_uint64x2x2_to_8xfloat32_nbits_DIC(input, output, guard_bit, false);
	for (int i = 0; i < 8; i++)
	{
		cout << "output_ref[" << i << "]=" << output_ref[i] << ", output[" << i << "]=" << output[i] << endl;
		if (output[i] != output_ref[i])
		{
			std::cout << "Test failed!" << std::endl;
			return false;
		}
	}
	std::cout << "Test pass!" << std::endl;
	return true;
}

// 用于验证结果是否正确
template <class T>
void hipack_naive_imp(const T *inp_ptr, const T *weight_ptr, float *output_ptr,
					  int N, int Ci, int H, int W, int Co, int K, int Ho, int Wo, int padding = 0,
					  int a_bit = 4, int w_bit = 2, int ar_bit = 32, int wr_bit = 32,
					  int stride = 1, int dilation = 1)
{

	// int interval = K;
	// int guard_bit = (a_bit+w_bit)+ceil(log(float(K))/log(2.));
	int guard_bit = 13; // using the 8-bit to split the results as uchar.
	if (a_bit >= 7 || w_bit >= 7)
	{
		guard_bit = 12;
	}

	union half_hiconv_packed_out
	{
		/* data */
		uint32 u32_packed_value;
		// uint8 u8_value[4];
	};
	union hiconv_packed_out
	{
		/* data */
		uint64 u64_packed_value;
		// uint32 u32_packed_value[2];
		// uint8 u8_value[8];
	};

// compute output
#ifdef ENABLE_OPENMP
	omp_set_num_threads(4);
#pragma omp parallel for collapse(2)
#endif
	for (int x = 0; x < N; x++)
	{
		for (int y = 0; y < Co; y++)
		{
			// half_hiconv_packed_out pre_half_packed_output;
			// pre_half_packed_output.u32_packed_value = 0;
			int H_upper = H;
			for (int h = -(K - 1); h < H_upper; h++)
			{
				for (int z = 0; z < W; z += K)
				{ // K is the interval
					int kh_low = h < 0 ? (-h) : 0;
					int kh_upper = (H_upper - h > K) ? K : (H_upper - h);
					int output_idx = x * Co * Ho * Wo + y * Ho * Wo + (h + (K - 1)) * Wo + z;
					for (int i = 0; i < Ci; i++)
					{
						for (int kh = kh_low; kh < kh_upper; kh++)
						{
							// packing the fearture in one uint32 variable and then covert to float32 variable
							uint32 packed_inp = 0, packed_weight = 0;
							hiconv_packed_out packed_output;
							packed_output.u64_packed_value = 0;
							// packing
							for (int k = 0; k < K; k++)
							{
								int posw = int(z * stride + k * dilation);
								int posh = int(h * stride + kh * dilation);
								int weight_idx = y * Ci * K * K + i * K * K + kh * K + k;
								int inp_idx = x * Ci * H * W + i * H * W + posh * W + posw;
								if (posh < H && posw < W)
									packed_inp += (unsigned int)inp_ptr[inp_idx] << (guard_bit * k);
								packed_weight += (unsigned int)weight_ptr[weight_idx] << (guard_bit * (K - k - 1));
							}
							// computation
							packed_output.u64_packed_value = (uint64)packed_weight * (uint64)packed_inp; // + pre_half_packed_output.u32_packed_value;
							// split
							for (int k = 0; k < K + 2; k++)
							{
								int output_value = (packed_output.u64_packed_value >> (guard_bit * k)) & ((1 << guard_bit) - 1);
								output_ptr[output_idx + k] += (float)output_value; // packed_output.u8_value[k];
							}
						}
					}
				}
			}
		}
	}
}

// 根据WA_bits以及MT的参数找到最合适的执行函数体，可以用指针来做
using FnPtr = void (*)(const int *, const int *, float *,
					   int, int, int, int, int, int, int, int, int,
					   int, int, int, int,
					   int, int);

bool test_bench(int N,
				int Ci, int H, int W, int Co, int W_bits = 3, int A_bits = 3, // int K = 3,
				bool DEBUG = false, bool verbose = false, FnPtr direct_conv2d_func = nullptr)
{
	// generate test data
	const int repeat = 5;
	int K = 3;
	int stride = 1, dilation = 1, padding = 0, margin = (K + 1) / 2;

	int amin = 0, amax = pow(2, A_bits) - 1; // 2**4-1
	int wmin = 0, wmax = pow(2, W_bits) - 1; // 2**2-1

	const int TN = 2; // 2路并行计算
	const int regA = 2;
	const int regW = 2;
	const int regN = 1;
	int align_num = regA * K * TN;
	// int align_num = TN;

	// int _H = (H/K)*K+((H%K)>0)*K; // W必须三位对齐，否则会出现访存越界。
	// H = (H/align_num)*align_num+((H%align_num)>0)*align_num;
	W = (W / align_num) * align_num + ((W % align_num) > 0) * align_num;
	N = (N / regN) * regN + ((N % regN) > 0) * regN;

	int Ho_base = int((H + 2 * padding - (K + (K - 1) * (dilation - 1))) / stride) + 1;
	int Wo_base = int((W + 2 * padding - (K + (K - 1) * (dilation - 1))) / stride) + 1 + 2 * margin;
	// hiconv会more计算最大的输出，因为其直接一步就可以计算出，为了方便，就直接在输出上加上一圈
	int _Ho = int((H + 2 * padding - (K + (K - 1) * (dilation - 1))) / stride) + 1 + 2 * margin;
	// int _Ho = H-2;
	int _Wo = int((W + 2 * padding - (K + (K - 1) * (dilation - 1))) / stride) + 1 + 2 * margin;

	// std::cout << " Ho_base=" << Ho_base << " Wo_base=" << Wo_base << " _Ho=" << _Ho << " _Wo=" << _Wo << std::endl;

	double hiconv_FLOPs = double(N) * double(Ci) * double(_Ho) * double(_Wo) * double(Co) * K * K * 2;
	// double hiconv_FLOPs = double(N) * double(Ci) * double(_Ho - 2 * margin) * double(_Wo) * double(Co) * K * K * 2;

	// typedef unsigned int T;
	const int align_bits = 128;

	int *inp_ptr = static_cast<int *>(std::aligned_alloc(align_bits, N * Ci * H * W * sizeof(int)));
	int *weight_ptr = static_cast<int *>(std::aligned_alloc(align_bits, Co * Ci * K * K * sizeof(int)));

	float *output_ptr_base = static_cast<float *>(std::aligned_alloc(align_bits, N * Co * _Ho * _Wo * sizeof(float)));

	float *output_ptr = static_cast<float *>(std::aligned_alloc(align_bits, N * Co * _Ho * _Wo * sizeof(float)));

	// float *output_ptr = output_ptr_raw;

	// initialize the input
	for (int i = 0; i < N * Ci * H * W; i++)
	{
		inp_ptr[i] = getRand(amin, amax);
	}
	if (DEBUG && verbose)
	{
		std::cout << "inp=[" << std::endl;
		for (int i = 0; i < N * Ci; i++)
		{
			for (int kh = 0; kh < H; kh++)
			{
				for (int kw = 0; kw < W; kw++)
				{
					std::cout << std::setw(4) << inp_ptr[i * (H * W) + kh * W + kw] << ",";
				}
				std::cout << std::endl;
			}
			if (i < N * Ci - 1)
				std::cout << std::endl;
			else
				std::cout << "]" << std::endl
						  << std::endl;
		}
	}

	// initialize the weight
	for (int i = 0; i < Co * Ci * K * K; i++)
	{
		weight_ptr[i] = getRand(wmin, wmax);
	}
	if (DEBUG && verbose)
	{
		std::cout << "weight=[" << std::endl;
		for (int i = 0; i < Co * Ci; i++)
		{
			for (int kh = 0; kh < K; kh++)
			{
				for (int kw = 0; kw < K; kw++)
				{
					std::cout << std::setw(4) << weight_ptr[i * (K * K) + kh * K + kw] << ",";
				}
				std::cout << std::endl;
			}
			if (i < Co * Ci - 1)
				std::cout << std::endl;
			else
				std::cout << "]" << std::endl
						  << std::endl;
		}
	}

	for (int i = 0; i < N * Co * Ho_base * Wo_base; i++)
	{
		output_ptr_base[i] = 0;
	}
	for (int i = 0; i < N * Co * _Ho * _Wo; i++)
	{
		output_ptr[i] = 0;
	}

	// warmup
	for (int i = 0; i < 1; i++)
	{
		if (DEBUG)
		{
			hipack_naive_imp(inp_ptr, weight_ptr, output_ptr_base,
							 N, Ci, H, W, Co, K, _Ho, _Wo, padding, A_bits, W_bits, 32, 32, 1, 1);
			if (verbose)
			{
				std::cout << "True Output=[" << std::endl;
				for (int i = 0; i < N * Co; i++)
				{
					for (int kh = 0; kh < _Ho; kh++)
					{
						for (int kw = 0; kw < _Wo; kw++)
						{
							std::cout << std::setw(4) << output_ptr_base[i * (_Ho * _Wo) + kh * _Wo + kw] << ",";
						}
						std::cout << std::endl;
					}
					if (i < N * Co - 1)
						std::cout << std::endl;
					else
						std::cout << "]" << std::endl
								  << std::endl;
				}
			}
		}
		direct_conv2d_func(inp_ptr, weight_ptr, output_ptr,
						   N, Ci, H, W, Co, K, _Ho, _Wo, padding, A_bits, W_bits, 32, 32, 1, 1);
		if (DEBUG && verbose)
		{
			std::cout << "HIPACK Output=[" << std::endl;
			for (int i = 0; i < N * Co; i++)
			{
				for (int kh = 0; kh < _Ho; kh++)
				{
					for (int kw = 0; kw < _Wo; kw++)
					{
						std::cout << std::setw(4) << output_ptr[i * (_Ho * _Wo) + kh * _Wo + kw] << ",";
					}
					std::cout << std::endl;
				}
				if (i < N * Co - 1)
					std::cout << std::endl;
				else
					std::cout << "]" << std::endl
							  << std::endl;
			}
		}
	}

	// Test the correctness of function
	if (DEBUG)
	{
		bool pass = true;
		for (int i = 0; i < N * Co; i++)
		{
			for (int kh = 0; kh < _Ho; kh++)
			{
				for (int kw = 0; kw < _Wo; kw++)
				{
					// std::cout << std::setw(4) << output_ptr[kh * _Wo + kw] << ",";
					// Test the two value
					float v1 = output_ptr_base[i * _Ho * _Wo + kh * _Wo + kw];
					float v2 = output_ptr[i * _Ho * _Wo + kh * _Wo + kw];
					// assert(v1 == v2);
					if (v1 != v2)
					{
						pass = false;
						std::cout << "At position [" << i << ", " << kh << ", " << kw << "]: " << v1 << " != " << v2 << " ";
						// std::cout << failed_words << std::endl;
						break;
					}
				}
			}
		}
		printf("\t[W%dA%d] input[%d,%d,%d,%d] * weight[%d,%d,%d,%d]: ",
			   W_bits, A_bits, N, Ci, H, W, Co, Ci, K, K);
		if (pass)
		{
			std::cout << passed_words << std::endl;
		}
		else
		{
			std::cout << failed_words << std::endl;
		}
	}

	// Test the performance of function
	auto start = std::chrono::system_clock::now();
	for (int i = 0; i < repeat; i++)
	{
		direct_conv2d_func(inp_ptr, weight_ptr, output_ptr,
						   N, Ci, H, W, Co, K, _Ho, _Wo, padding, A_bits, W_bits, 32, 32, 1, 1);
	}
	auto end = std::chrono::system_clock::now();
	double hiconv_time = getTime(start, end) / repeat;
	double hiconv_perf = hiconv_FLOPs / hiconv_time / 1e9;

	printf("\t[W%dA%d] input[%d,%d,%d,%d] * weight[%d,%d,%d,%d]: Elapsed time: %f seconds Performance: %f GFLOPS.\n",
		   W_bits, A_bits, N, Ci, H, W, Co, Ci, K, K,
		   hiconv_time, hiconv_perf);

	std::free(inp_ptr);
	std::free(weight_ptr);
	std::free(output_ptr_base);
	std::free(output_ptr);

	return true;
}