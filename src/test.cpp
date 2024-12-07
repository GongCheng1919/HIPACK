#include "base.h"
#include <stdio.h>
#include <cassert>
int main()
{

	// Test 1
	printf("Test 1: test the vld2q_u64 loading order\n");
	// 定义一个包含 4 个 uint64_t 元素的数组
	uint64_t data[4] = {1, 2, 3, 4};

	// 使用 vld2q_u64 将数组交错加载到 uint64x2x2_t 类型的数据中
	uint64x2x2_t result = vld2q_u64(data);

	// 打印结果
	printf("result.val[0][0] = %llu\n", result.val[0][0]);
	printf("result.val[0][1] = %llu\n", result.val[0][1]);
	printf("result.val[1][0] = %llu\n", result.val[1][0]);
	printf("result.val[1][1] = %llu\n", result.val[1][1]);

	// Test 2: DIC
	assert(Test_accumlate_uint64x2x2_to_8xfloat32_nbits_DIC());
	// Test 3: 逻辑右移
	printf("Test 3: logical right shift\n");
	// 定义一个 uint64x2_t 向量
	uint64x2_t vec = {0xeFFFFFFFFFFFFFF0, 0x123456789ABCDEF0};

	// 定义右移的位数
	int m = 4;

	// 对向量中的每个元素进行逻辑右移 m 位
	uint64x2_t result2 = vshrq_n_u64(vec, m);

	// 打印结果
	printf("Original vec[0] = %llx\n", vec[0]);
	printf("Original vec[1] = %llx\n", vec[1]);
	printf("Shifted result[0] = %llx\n", result2[0]);
	printf("Shifted result[1] = %llx\n", result2[1]);
	return 0;
}