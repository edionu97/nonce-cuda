#pragma once

//cuda specific
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//user defined
#include "../hasher/device_sha1_hasher.h"
#include "../cuda_helpers/cuda_utils.h"
#include "../helpers/nonce_helpers.h"
#include "../helpers/string_helpers.h"

namespace gpu_nonce_computation
{
	
	/**
	 * \brief This structure represents the data that will be passed to the device
	 */
	struct device_data
	{
		size_t number_of_words;

		//the prefix and suffix data
		const char* prefix_data;
		const char* suffix_data;

		//the array of words
		const char* word_array;

		//each word boundary
		const size_t* word_boundaries;

		
		/**
		 * \brief This represents the constructor of the structure
		 * \param prefix : the prefix
		 * \param suffix : the suffix
		 * \param word : the word
		 * \param boundaries : each word boundary (each value represents a number)
		 * \param word_number : the number of words
		 */
		explicit device_data(
			const char* prefix,
			const char* suffix,
			const char* word,
			const size_t* boundaries,
			const size_t word_number)
		{
			prefix_data = prefix;
			suffix_data = suffix;
			word_array = word;
			word_boundaries = boundaries;
			number_of_words = word_number;
		}
	};
	
	/**
	 * \brief This structure represents the out computation result
	 * Each thread will try to find the value, and if success the found value will be true and the word_index will indicate the index of the word
	 */
	struct out_thread_computation_result
	{
		bool found{ false };
		size_t word_index{};
	};

	/**
	 * \brief
	 * This function represents the kernel that takes care of nonce data_computation
	 * \param device_data : the structure that contains information about the data that will be processed
	 * \param out_data : the out data array vector
	 * \param thread_data_chunk : the number of elements that will be processed by gpu thread (by default this is 100)
	 */
	// ReSharper disable once CppNonInlineFunctionDefinitionInHeaderFile
	__global__ void compute_nonce(const device_data device_data,
	                              out_thread_computation_result* out_data,
	                              const size_t thread_data_chunk = 100)
	{
		//compute the global thread index
		const auto global_thread_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

		//check that we are in bounds
		//the if the thread can process it's stride (or at least a part of it) this condition must pass
		if (thread_data_chunk * global_thread_index >= device_data.number_of_words)
		{
			return;
		}

		const auto ends_with_size = static_cast<int>(string_helpers::device_strlen(device_data.suffix_data));

		//create a hasher instance
		device_sha1_hasher hasher{};

		//compute the context for the prefix
		hasher.update(device_data.prefix_data);

		char result[41]{};
		const auto prefix_context_copy = hasher.get_context_copy();

		//get the number of iterations
		const auto max_size = nonce_helpers::min(
			device_data.number_of_words,
			(global_thread_index + 1) * thread_data_chunk
		);

		//process thread chunk of data
		//thread 0 will process items from 0 to thread_data_chunk
		//thread 1 will process items from thread_data_chunk to 2 * thread_data_chunk and so on
		for (auto index = thread_data_chunk * global_thread_index; index < max_size; ++index)
		{
			//clear the result
			memset(result, NULL, sizeof(result));

			//set the context to the value before suffix computation
			hasher.set_context(prefix_context_copy);

			//add the suffix for message digest
			hasher.update(nonce_helpers::get_nth_word(index, device_data.word_array, device_data.word_boundaries));

			//get the final result
			hasher.get_final(result);

			//if the result does not end with desired suffix skip the generate_nonce_alphabet
			const auto result_length = static_cast<int>(string_helpers::device_strlen(result));
			if (!string_helpers::ends_with(result, device_data.suffix_data, result_length, ends_with_size))
			{
				continue;
			}

			//set the proper generate_nonce_alphabet (mark that we found  the value and return from function)
			out_data[global_thread_index].found = true;
			out_data[global_thread_index].word_index = index;
			
			return;
		}
	}
	
}
