#include <queue>
#include <iostream>

#include "cuda_helpers/cuda_data_convertor.h"
#include "cuda_helpers/cuda_device_helpers.h"
#include "cuda_helpers/cuda_utils.h"
#include "helpers/nonce_helpers.h"

#include "gpu/nonce.cuh"


#include "hasher/host_sha1_hasher.h"
#include "resources/manager/resources_manager.h"

using namespace  gpu_nonce_computation;

std::ostream& operator << (std::ostream& out, const std::pair<std::string, std::string>& value)
{
	out << "\n\nNonce found\n";
	out << "Nonce value: " << value.first << '\n';
	out << "Computed sha1 for nonce: " << value.second << '\n';

	return out;
}

class nonce
{
public:

	nonce(const configuration& configuration)
		:configuration_{ configuration }
	{
		//initialize the alphabet 
		from_values_ = generate_nonce_alphabet();
	}

	/**
	 * \brief
	 * This function it is used for computing the nonce value
	 * \param nonce_prefix : the prefix
	 * \param sha_suffix : the value in which the suffix must end
	 * \return a pair (first element is the the word nonce, and the second one is the sha1 value)
	 */
	std::pair<std::string, std::string> get_nonce(const std::string& nonce_prefix, const std::string& sha_suffix)
	{
		return generate_possible_suffixes(nonce_prefix, sha_suffix);
	}

private:

	/**
	 * \brief
	 * This function generates the values that are used for nonce suffix generation
	 * \return An array of characters
	 */
	static std::vector<char> generate_nonce_alphabet()
	{
		//generate the nonce alphabet
		std::vector<char> character_array{};
		for (uint8_t character = 'a'; character < 'z' + 1; ++character)
		{
			character_array.push_back(static_cast<char>(character));
		}

		for (uint8_t character = 'A'; character < 'Z' + 1; ++character)
		{
			character_array.push_back(static_cast<char>(character));
		}

		for (uint8_t character = '0'; character < '9' + 1; ++character)
		{
			character_array.push_back(static_cast<char>(character));
		}

		return  character_array;
	}

	/**
	 * \brief
	 * This method generates a chunk of possible suffixes for nonce
	 */
	std::pair<std::string, std::string> generate_possible_suffixes(const std::string& nonce_prefix, const std::string& sha_suffix)
	{
		//initialize the queue
		std::queue<std::string> solution_queue{};
		for (const auto chr : from_values_)
		{
			solution_queue.push(std::string{ chr });
		}

		//generate the other values in ordered form
		std::vector<std::string> solution_chunk{ };
		while (!solution_queue.empty())
		{
			//get the item from queue
			const auto& current_solution = solution_queue.front();

			//push the current solution in array
			solution_chunk.push_back(current_solution);

			//if we have a complete chunk of data than process it
			if (solution_chunk.size() == configuration_.cpu_generated_chunk_size)
			{
				//get the result
				const auto result = process_solution_chunk(solution_chunk, nonce_prefix, sha_suffix);

				if (result.first)
				{
					return  get_result(result, nonce_prefix);
				}

				std::cout << "Processed: " << solution_chunk.size() << " elements, no nonce found, keep trying...\n\n";
				solution_chunk = std::vector<std::string>();
			}

			//add the other values
			for (const auto chr : from_values_)
			{
				solution_queue.push(current_solution + chr);
			}

			//remove the values from queue
			solution_queue.pop();
		}

		const auto result = process_solution_chunk(solution_chunk, nonce_prefix, sha_suffix);
		if (result.first)
		{
			return  get_result(result, nonce_prefix);
		}

		return { "Not found", "-" };
	}

	/**
	 * \brief
	 * This method it is used for processing the solution chunk
	 * \param solution_chunks : a chunk of data that represents the possible suffixes of data
	 * \param nonce_prefix : the prefix
	 * \param sha_suffix : the sha suffix
	 */
	std::pair<bool, std::string> process_solution_chunk(const std::vector<std::string>& solution_chunks, const std::string& nonce_prefix, const std::string& sha_suffix) const
	{
		//if we have 0 elements to process than ignore the values
		if (solution_chunks.empty())
		{
			return { false, "" };
		}

		statistics::elapsed_time_computer time_computer;

		try
		{

			//set the beginning of the time measurement
			time_computer.set_time_period();

			//liniearize the matrix
			const auto get_linearized_matrix = nonce_helpers::liniearize_word_matrix_array(solution_chunks);

			//liniearize the prefix and the suffix
			const auto prefix_linearized = nonce_helpers::liniearize_word_matrix_array({ nonce_prefix });
			const auto suffix_linearized = nonce_helpers::liniearize_word_matrix_array({ sha_suffix });

			//allocate the array for output data
			const auto out_vector = std::vector<out_thread_computation_result>(solution_chunks.size(), out_thread_computation_result{});
			const auto out_data = cuda_data_convertor<out_thread_computation_result>::convert_stl_vector_to_device_array(out_vector);

			//get the device data for the prefix and suffix
			const auto prefix_data = cuda_data_convertor<char>::convert_stl_vector_to_device_array(prefix_linearized.second);
			const auto suffix_data = cuda_data_convertor<char>::convert_stl_vector_to_device_array(suffix_linearized.second);

			//get the device information for word boundary
			const auto words_data = cuda_data_convertor<char>::convert_stl_vector_to_device_array(get_linearized_matrix.second);
			const auto boundaries_data = cuda_data_convertor<size_t>::convert_stl_vector_to_device_array(get_linearized_matrix.first);

			//set the data transfer period (cpu to gpu)
			time_computer.set_time_period("\tTransfer data from cpu to gpu took: ");

			//instantiate the device data
			const device_data kernel_information{
			prefix_data.get_device_data(),
			suffix_data.get_device_data(),
			words_data.get_device_data(),
			boundaries_data.get_device_data(),
			solution_chunks.size() };

			//get the launch configuration
			const auto launch_config = get_launch_parameters(solution_chunks);

			printf("\nLaunch kernel (%d blocks of %d threads) for computing sha1 against %llu elements\n", launch_config.first.x, launch_config.second.x, solution_chunks.size());

			//launck the kernel
			compute_nonce << <launch_config.first, launch_config.second >> > (kernel_information, out_data.get_device_data(), configuration_.thread_chunks_number);

			//set the kernel period 
			time_computer.set_time_period("\tKernel processing took: ");

			//get the results vector back
			const auto results = cuda_data_convertor<out_thread_computation_result>::convert_device_data_to_stl_vector(out_data, out_vector.size());

			//check for error
			cuda_device_helpers::check();

			//mark the last event
			time_computer.set_time_period("\tTransfer data back from gpu to cpu took: ", true);

			//check if we found something
			const auto solution_position = std::find_if(
				results.begin(),
				results.end(),
				[](const out_thread_computation_result& data)
				{
					return data.found;
				});

			//print time periods
			time_computer.print_time_periods(std::cout);

			//check for error
			cuda_device_helpers::check();

			//if no solution is found than return nothing
			if (solution_position == results.end())
			{
				return { false, "" };
			}

			//get the solution
			return { true, solution_chunks[solution_position->word_index] };
		}
		catch (std::exception& e)
		{
			std::cout << e.what() << '\n';
		}

		return { false, "" };
	}

	std::pair<dim3, dim3> get_launch_parameters(const std::vector<std::string>& solution_chunks) const
	{
		//get the number of items that will be processed by each block of threads
		const auto suffixes_processed_by_block = configuration_.max_threads_per_block * configuration_.thread_chunks_number;

		//get the number of blocks
		const auto block_numbers = static_cast<unsigned int>(ceil(static_cast<double>(solution_chunks.size()) / static_cast<double>(suffixes_processed_by_block)));

		//get the number of blocks and threads
		return  std::make_pair<dim3, dim3>(dim3{ block_numbers }, dim3{ configuration_.max_threads_per_block });
	}

	std::pair<std::string, std::string> get_result(const std::pair<bool, std::string>& computation_result, const std::string& prefix)
	{
		if (!computation_result.first)
		{
			return  {};
		}

		std::pair<std::string, std::string> nonce = {
			prefix + computation_result.second,
			sha1_hasher_.compute_multiple_sha1(prefix, { computation_result.second }).back()
		};

		std::cout << nonce << '\n';
		return nonce;
	}


	std::vector<char> from_values_;

	host_sha1_hasher sha1_hasher_;

	const configuration& configuration_;
};

using namespace cuda_print_utils;

// prefix : e, sufix eebdc

//Nonce value : ea6K5 sha1 is 8699397b07444a59ad7a0dcf478110026d0eebdc

auto main() -> int  // NOLINT(bugprone-exception-escape)
{
	try
	{

		//read the configuration
		const auto config = resources_manager::get_config();

		//get the cuda device interaction
		const auto cuda_device_interaction = cuda_device_helpers::get_cuda_device_interactions();

		//check if we have installed some nvidia gpu
		if (cuda_device_interaction.get_number_of_installed_devices() == 0)
		{
			throw std::exception("No Nvidia Gpu installed");
		}

		//get device properties
		std::cout << cuda_device_interaction.get_device_property(config.device_name) << '\n';

		nonce nonce_computer{ config };

		std::cout << "Nonce prefix: " << config.prefix << '\n';
		std::cout << "Desired suffix: " << config.suffix << '\n';

		//get the nonce
		nonce_computer.get_nonce(config.prefix, config.suffix);
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << '\n';
	}

	return 0;
}

