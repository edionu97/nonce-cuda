
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <iostream>


#include "hasher/sha1_hasher.h"
#include "helpers/string_helpers.h"


__global__ void kernel()
{
	sha1_hasher hasher{};

	hasher.update("g");
	hasher.update("r");
	hasher.update("a");
	hasher.update("p");
	hasher.update("e");

	char result[41];
	memset(result, 0, sizeof(result));

	hasher.get_final(result);

	//printf("%s\n", result);

}

int main()
{
	kernel << <50, 512 >> > ();

	cudaDeviceSynchronize();

	std::cout << cudaGetErrorString(cudaGetLastError()) << '\n';

}

