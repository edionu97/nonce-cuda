#include <iostream>

#include "cuda_helpers/cuda_data_convertor.h"
#include "cuda_helpers/cuda_device_helpers.h"
#include "cuda_helpers/cuda_utils.h"
#include "helpers/nonce_helpers.h"

#include "gpu/nonce.cuh"

using namespace  gpu_nonce_computation;



int main()
{

	try
	{
		std::vector<std::string> words{ "ape", "andma", "ape", "een", "dddadadbsdbsbdhusdbhsjbdhbhsbdhsbdhsbdhbsh" };

		for (int i = 0; i < 50000; ++i)
		{
			words.emplace_back("ape");
		}

		words.emplace_back("adle");


		const auto linearized = nonce_helpers::liniearize_word_matrix_array(words);

		const auto prefix_linearized = nonce_helpers::liniearize_word_matrix_array({ "gr" });
		const auto suffix_linearized = nonce_helpers::liniearize_word_matrix_array({ "76a" });

		const auto prefix_data = cuda_data_convertor<char>::convert_stl_vector_to_device_array(prefix_linearized.second);
		const auto suffix_data = cuda_data_convertor<char>::convert_stl_vector_to_device_array(suffix_linearized.second);


		const auto words_data = cuda_data_convertor<char>::convert_stl_vector_to_device_array(linearized.second);
		const auto boundaries_data = cuda_data_convertor<size_t>::convert_stl_vector_to_device_array(linearized.first);

		const auto out_vector = std::vector<out_thread_computation_result>(1 * 512, gpu_nonce_computation::out_thread_computation_result{});
		const auto out_data = cuda_data_convertor<out_thread_computation_result>::convert_stl_vector_to_device_array(out_vector);

		const device_data d_data{
			prefix_data.get_device_data(),
			suffix_data.get_device_data(),
			words_data.get_device_data(),
			boundaries_data.get_device_data(),
			words.size() };

		compute_nonce << <1, 512 >> > (d_data, out_data.get_device_data());

		const auto vector = cuda_data_convertor<out_thread_computation_result>::convert_device_data_to_stl_vector(out_data, out_vector.size());

		const auto it = std::find_if(vector.begin(), vector.end(), [](const out_thread_computation_result& data)
			{
				return data.found;
			});


		cuda_device_helpers::check();

		if (it == vector.end())
		{
			std::cout << "No data found";
			return 0;
		}

		std::cout << words[it->word_index] << '\n';
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << '\n';
	}



}

