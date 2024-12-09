#include <torch/extension.h>
#include "hipack_conv2d.h"
#include <vector>
#include <iostream>
#include <chrono>
// #include "utils.h"

typedef int T;

// Wrapper function
torch::Tensor direct_conv2d(torch::Tensor &inp, torch::Tensor &weight,
                            int W_bits, int A_bits, bool MT = true,
                            int stride = 1, bool show_time = false,
                            bool depth_conv = false
                            // int padding = 0, int dilation = 1
)
{
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    std::chrono::duration<double> elapsed;

    if (show_time)
        start = std::chrono::high_resolution_clock::now();
    // Convert tensors to contiguous and get raw pointers
    inp = inp.contiguous();
    weight = weight.contiguous();
    T *inp_ptr = inp.data_ptr<T>();
    T *weight_ptr = weight.data_ptr<T>();
    int N = inp.size(0);
    int Ci = inp.size(1);
    int H = inp.size(2);
    int W = inp.size(3);
    int Co = weight.size(0);
    int K = weight.size(2);
    // stride 已经被设定了，不会改变
    int dilation = 1, padding = 0, margin = (K + 1) / 2;

    // int TN = 2; // 2路并行计算
    // if (W_bits + A_bits < 3)
    // {
    //     TN = 4;
    // }
    const int regA = 2;
    // int align_num = regA * K * TN;
    // int align_num = TN * K;
    int align_num = 6;

    // int Ho = int((H+2*padding-(K+(K-1)*(dilation-1)))/stride) + 1;
    // int Wo = int((W+2*padding-(K+(K-1)*(dilation-1)))/stride) + 1;
    // hiconv会more计算最大的输出，因为其直接一步就可以计算出，为了方便，就直接在输出上加上一圈
    int _Ho = int((H + 2 * padding - (K + (K - 1) * (dilation - 1))) / stride) + 1 + 2 * margin;
    int _Wo = int((W + 2 * padding - (K + (K - 1) * (dilation - 1))) / stride) + 1 + 2 * margin;

    // Allocate output tensor
    torch::Tensor output = torch::empty({N, Co, _Ho, _Wo}, inp.options().dtype(torch::kFloat));
    float *output_ptr = output.data_ptr<float>();

    // get function pointer
    // 根据WA_bits以及MT的参数找到最合适的执行函数体，可以用指针来做
    using FnPtr = void (*)(const T *, const T *, float *,
                           int, int, int, int, int, int, int, int, int,
                           int, int, int, int,
                           int, int);
    FnPtr direct_conv2d_func = hipack_conv2d_v3;

    // if ((W_bits == 1 && A_bits <= 3) || (A_bits == 1 && W_bits <= 3))
    // if ((W_bits + A_bits) <= 4)
    // {
    //     if (depth_conv)
    //     {
    //         direct_conv2d_func = depth_hiconv2d_simd_complete_7bits_w1a1_inputrepackingv2_nofloat32x4caching_addint16x8caching_imp<T, 4, 8, 2, 1>;
    //     }
    //     else
    //     {
    //         if (MT)
    //         {
    //             direct_conv2d_func = hiconv2d_simd_complete_7bits_w1a1_inputrepackingv2_nofloat32x4caching_addint16x8caching_imp<T, 4, 8, 2, 1>;
    //         }
    //         else
    //         {
    //             direct_conv2d_func = hiconv2d_simd_complete_7bits_w1a1_inputrepackingv2_nofloat32x4caching_addint16x8caching_imp_ST<T, 4, 8, 2, 1>;
    //         }
    //     }
    // }
    // else
    // {
    //     if (depth_conv)
    //     {
    //         direct_conv2d_func = depth_hiconv2d_simd_complete_14bits_inputrepackingv2_nofloat32x4caching_addint16x8caching_imp<T, 4, 8, 2, 1>;
    //     }
    //     else
    //     {
    //         if (MT)
    //         {
    //             direct_conv2d_func = hiconv2d_simd_complete_14bits_inputrepackingv2_nofloat32x4caching_addint16x8caching_imp<T, 4, 8, 2, 1>;
    //         }
    //         else
    //         {
    //             direct_conv2d_func = hiconv2d_simd_complete_14bits_inputrepackingv2_nofloat32x4caching_addint16x8caching_imp_ST<T, 1, 2, 2, 1>;
    //         }
    //     }
    // }
    if (show_time)
    {
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "Preprocess time: " << elapsed.count() << "s\n";
    }

    // const int align_bits = 128;
    // inp_ptr = static_cast<T *>(std::aligned_alloc(align_bits, N * Ci * H * W * sizeof(T)));
    // weight_ptr = static_cast<T *>(std::aligned_alloc(align_bits, Co * Ci * K * K * sizeof(T)));
    // output_ptr = static_cast<float *>(std::aligned_alloc(align_bits, N * Co * _Ho * _Wo * sizeof(float)));
    // // initialize the input
    // int amin = 0, amax = pow(2, A_bits) - 1; // 2**4-1
    // int wmin = 0, wmax = pow(2, W_bits) - 1; // 2**2-1
    // for (int i = 0; i < N * Ci * H * W; i++)
    // {
    //     inp_ptr[i] = getRand(amin, amax);
    // }
    // // initialize the weight
    // for (int i = 0; i < Co * Ci * K * K; i++)
    // {
    //     weight_ptr[i] = getRand(wmin, wmax);
    // }
    // for (int i = 0; i < N * Co * _Ho * _Wo; i++)
    // {
    //     output_ptr[i] = 0;
    // }

    // Call your function
    // 先运行几次以避免初次载入出错
    // for (int i = 0; i < 4; i++)
    //     direct_conv2d_func(inp_ptr, weight_ptr, output_ptr,
    //                        N, Ci, H, W, Co, K, _Ho, _Wo, padding, A_bits, W_bits, 32, 32, 1, 1);

    // 计时开始
    if (show_time)
        start = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < 4; i++)
    direct_conv2d_func(inp_ptr, weight_ptr, output_ptr,
                       N, Ci, H, W, Co, K, _Ho, _Wo, padding, A_bits, W_bits, 32, 32, stride, 1);
    // 计时结束
    if (show_time)
    {
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "direct_conv2d computation time: " << elapsed.count() << "s\n";
    }
    return output;
}

// Wrapper function
torch::Tensor direct_conv2d_khkw(torch::Tensor &inp, torch::Tensor &weight,
                                 int W_bits, int A_bits, bool MT = true,
                                 int stride = 1, bool show_time = false,
                                 bool depth_conv = false
                                 // int padding = 0, int dilation = 1
)
{
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    std::chrono::duration<double> elapsed;

    if (show_time)
        start = std::chrono::high_resolution_clock::now();
    // Convert tensors to contiguous and get raw pointers
    inp = inp.contiguous();
    weight = weight.contiguous();
    T *inp_ptr = inp.data_ptr<T>();
    T *weight_ptr = weight.data_ptr<T>();
    int N = inp.size(0);
    int Ci = inp.size(1);
    int H = inp.size(2);
    int W = inp.size(3);
    int Co = weight.size(0);
    int KH = weight.size(2);
    int KW = weight.size(3);
    // std::cout << "KH:" << KH << ",KW:" << KW << std::endl;
    // stride 已经被设定了，不会改变
    int dilation = 1, padding = 0, marginH = (KH + 1) / 2, marginW = (KW + 1) / 2;
    int paddingH = (KH - 1);

    // int TN = 2; // 2路并行计算
    // if (W_bits + A_bits < 3)
    // {
    //     TN = 4;
    // }
    const int regA = 2;
    // int align_num = regA * K * TN;
    // int align_num = TN * K;
    int align_num = 6;

    // int Ho = int((H+2*padding-(K+(K-1)*(dilation-1)))/stride) + 1;
    // int Wo = int((W+2*padding-(K+(K-1)*(dilation-1)))/stride) + 1;
    // hiconv会more计算最大的输出，因为其直接一步就可以计算出，为了方便，就直接在输出上加上一圈
    int _Ho = int((H + 2 * paddingH - (KH + (KH - 1) * (dilation - 1))) / stride) + 1;
    int _Wo = int((W + 2 * padding - (KW + (KW - 1) * (dilation - 1))) / stride) + 1 + 2 * marginW;

    // Allocate output tensor
    torch::Tensor output = torch::empty({N, Co, _Ho, _Wo}, inp.options().dtype(torch::kFloat));
    float *output_ptr = output.data_ptr<float>();
    // std::cout << "_Ho:" << _Ho << ",_Wo:" << _Wo << std::endl;
    // get function pointer
    // 根据WA_bits以及MT的参数找到最合适的执行函数体，可以用指针来做
    using FnPtr = void (*)(const T *, const T *, float *,
                           int, int, int, int, int, int, int, int, int,
                           int, int, int, int, int,
                           int, int);
    FnPtr direct_conv2d_khkw_func = hipack_conv2d_khkw;

    // if ((W_bits == 1 && A_bits <= 3) || (A_bits == 1 && W_bits <= 3))
    // if ((W_bits + A_bits) <= 4)
    // {
    //     if (depth_conv)
    //     {
    //         direct_conv2d_func = depth_hiconv2d_simd_complete_7bits_w1a1_inputrepackingv2_nofloat32x4caching_addint16x8caching_imp<T, 4, 8, 2, 1>;
    //     }
    //     else
    //     {
    //         if (MT)
    //         {
    //             direct_conv2d_func = hiconv2d_simd_complete_7bits_w1a1_inputrepackingv2_nofloat32x4caching_addint16x8caching_imp<T, 4, 8, 2, 1>;
    //         }
    //         else
    //         {
    //             direct_conv2d_func = hiconv2d_simd_complete_7bits_w1a1_inputrepackingv2_nofloat32x4caching_addint16x8caching_imp_ST<T, 4, 8, 2, 1>;
    //         }
    //     }
    // }
    // else
    // {
    //     if (depth_conv)
    //     {
    //         direct_conv2d_func = depth_hiconv2d_simd_complete_14bits_inputrepackingv2_nofloat32x4caching_addint16x8caching_imp<T, 4, 8, 2, 1>;
    //     }
    //     else
    //     {
    //         if (MT)
    //         {
    //             direct_conv2d_func = hiconv2d_simd_complete_14bits_inputrepackingv2_nofloat32x4caching_addint16x8caching_imp<T, 4, 8, 2, 1>;
    //         }
    //         else
    //         {
    //             direct_conv2d_func = hiconv2d_simd_complete_14bits_inputrepackingv2_nofloat32x4caching_addint16x8caching_imp_ST<T, 1, 2, 2, 1>;
    //         }
    //     }
    // }
    if (show_time)
    {
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "Preprocess time: " << elapsed.count() << "s\n";
    }

    // const int align_bits = 128;
    // inp_ptr = static_cast<T *>(std::aligned_alloc(align_bits, N * Ci * H * W * sizeof(T)));
    // weight_ptr = static_cast<T *>(std::aligned_alloc(align_bits, Co * Ci * K * K * sizeof(T)));
    // output_ptr = static_cast<float *>(std::aligned_alloc(align_bits, N * Co * _Ho * _Wo * sizeof(float)));
    // // initialize the input
    // int amin = 0, amax = pow(2, A_bits) - 1; // 2**4-1
    // int wmin = 0, wmax = pow(2, W_bits) - 1; // 2**2-1
    // for (int i = 0; i < N * Ci * H * W; i++)
    // {
    //     inp_ptr[i] = getRand(amin, amax);
    // }
    // // initialize the weight
    // for (int i = 0; i < Co * Ci * K * K; i++)
    // {
    //     weight_ptr[i] = getRand(wmin, wmax);
    // }
    // for (int i = 0; i < N * Co * _Ho * _Wo; i++)
    // {
    //     output_ptr[i] = 0;
    // }

    // Call your function
    // 先运行几次以避免初次载入出错
    // for (int i = 0; i < 4; i++)
    //     direct_conv2d_func(inp_ptr, weight_ptr, output_ptr,
    //                        N, Ci, H, W, Co, K, _Ho, _Wo, padding, A_bits, W_bits, 32, 32, 1, 1);

    // 计时开始
    if (show_time)
        start = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < 4; i++)

    // std::cout << "Call functions" << _Wo << std::endl;
    direct_conv2d_khkw_func(inp_ptr, weight_ptr, output_ptr,
                            N, Ci, H, W, Co, KH, KW, _Ho, _Wo, padding, A_bits, W_bits, 32, 32, stride, 1);
    // 计时结束
    if (show_time)
    {
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "direct_conv2d computation time: " << elapsed.count() << "s\n";
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("direct_conv2d", &direct_conv2d, "HiConv2D SIMD complete input repacking v3 no float32x4 caching add int16x8 caching (CPU)");
    m.def("direct_conv2d_khkw", &direct_conv2d_khkw, "HiConv2D SIMD with varying kh kw version, complete input repacking v3 no float32x4 caching add int16x8 caching (CPU)");
}