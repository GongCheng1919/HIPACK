## version 0.0.0
The first version of directive convolution2d.
We employ data reorder, blocking, guard bit optimization, simd and parallisim optimization for high throughts.
Besides, we also employ the two-step int8x16 vector and dual int16x4x2 vector to cache the middle packed accumulating results for addressing depacking issues.

To do list:

- [ ] address the problems that high resolution input (espicially h-aixs) will seriously impact the computing performance.

## version 1.0.0
We have addressed high resolution input problem, which is caused by the invalid padding step, that pads a large pixels to input but do not compute the padding pixels, which cause a fake high computing performance.
In the version, we addressed the padding problems in direct conv2d. 
Specifically, we compute all the result with 2 padding pixel for conv2d computation, because direct conv2d natively compute padding results instead of a valid results. We may cause about 5GFLOPs decreas for computing performance.

To do list:

- [x] address the problems that high resolution input (espicially h-aixs) will seriously impact the computing performance.

- [ ] address any-resolution preblem in conv2d computation. Until now, the direct conv2d is efficient to compute conv2d operation with a regular input (divided by 6 or 12), which is inefficent when processing the irregular input such as resolution with 5x5, 7x7, 9x9。

- [ ] implement the pytorch function call.

## version 1.0.3
Modify the CXX compile args by inserting the custom ```cpp_extension._write_ninja_file_and_compile_objects``` as follows:
```python
raw_write_ninja_file_and_compile_objects = cpp_extension._write_ninja_file_and_compile_objects

def new_write_ninja_file_and_compile_objects(sources: List[str],
        objects,
        cflags,
        post_cflags,
        cuda_cflags,
        cuda_post_cflags,
        cuda_dlink_post_cflags,
        build_directory: str,
        verbose: bool,
        with_cuda: Optional[bool]):
    print("#########cflags: ",cflags)
    print("#########post_cflags: ",post_cflags)
    print("#########cuda_cflags: ",cuda_cflags)
    print("#########cuda_post_cflags: ",cuda_post_cflags)
    print("#########cuda_dlink_post_cflags: ",cuda_dlink_post_cflags)
    exclude_flags = ['-g', '-fwrapv',"-O2", "-pthread", "-Wno-unused-result", "-Wsign-compare"]
    # 如果pst_cflags中有-O1,-O2,-O3,-Ofast之类的标志，则去除-O2
    cflags = [arg for arg in cflags if arg not in exclude_flags]
    raw_write_ninja_file_and_compile_objects(sources, objects, cflags, post_cflags, cuda_cflags, cuda_post_cflags, cuda_dlink_post_cflags, build_directory, verbose, with_cuda)

cpp_extension._write_ninja_file_and_compile_objects = new_write_ninja_file_and_compile_objects
```

## version 1.0.4

We have modified the direct conv2d with W2A2. (Not implemented)

## version 1.0.5

We have modified the direct conv2d with W2A2.