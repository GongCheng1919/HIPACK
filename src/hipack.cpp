#include "hipack_conv2d.h"

int main(int argc, char **argv)
{
	// conv_float_testbench::test_bench();
	int N, Ci, H, W, Co, W_bits = 3, A_bits = 3;
	bool debug = true, verbose = true;
	if (argc >= 8)
	{
		// test_bench(argv[1],argv[2],argv[3],argv[4]);
		N = atoi(argv[1]);
		Ci = atoi(argv[2]);
		H = atoi(argv[3]);
		W = atoi(argv[4]);
		Co = atoi(argv[5]);
		W_bits = atoi(argv[6]);
		A_bits = atoi(argv[7]);
		// test_msb(N);
		// test_msb(Ci);
		// test_msb(Li);
		// test_msb(Co);
	}
	else
	{
		std::cout << "Usage: ./" << argv[0] << " N Ci H W Co[W_bits = 3 A_bits = 3][debug = true][verbose = true] " << std::endl;
		// std::cout << "default: ./main_aarch64 1 1 3 12 1 [W_bits=3 A_bits=3] [MT=false] [debug=true]" << std::endl;
		// return 1;
		N = 1;
		Ci = 1;
		H = 3;
		W = 12;
		Co = 1;
		W_bits = 3;
		A_bits = 3;
		debug = 1;
		verbose = 1;
	}
	if (argc >= 9)
	{
		debug = atoi(argv[8]);
	}
	if (argc >= 10)
	{
		verbose = atoi(argv[9]);
	}
	initRandomSeed();
	// test_convert_uint8x16_to_uint16x8();
	// test_bench(N, Ci, H, W, Co, W_bits, A_bits, MT, debug, hipack_conv2d);
	test_bench(N, Ci, H, W, Co, W_bits, A_bits, debug, verbose, hipack_conv2d_v3);
	return 0;
}