#pragma once
#include <string>
#include <nlohmann/json.hpp>
//
//{
//"cuda": {
//	"deviceName": "GeForce GTX 1070",
//		"maxThreadsPerBlock" : 128,
//		"threadChunksNumber" : 50
//},
//"nonce" : {
//		"prefix": "e",
//			"shaSuffix" : "ed"
//	},
//		"cpuGeneratedChunkSize" : 200000
//}

/// <summary>
/// Represents the json configuration mapped to an object
/// </summary>
struct configuration
{
	//cuda
	std::string device_name{};
	unsigned int max_threads_per_block{};
	size_t thread_chunks_number{};

	//cpu
	size_t cpu_generated_chunk_size{};

	//nonce
	std::string prefix{};
	std::string suffix{};

	/// <summary>
	/// Create the configuration from json aka parse the object
	/// </summary>
	/// <param name="json">the json object that contains data</param>
	explicit configuration(const nlohmann::json& json)
	{
		//parse cuda part
		const auto& cuda_json = json[cuda_tag_];
		device_name = cuda_json[device_name_tag_].get<std::string>();
		max_threads_per_block = cuda_json[max_threads_per_block_tag_].get<unsigned int>();
		thread_chunks_number = cuda_json[threads_chunks_number_tag_].get<size_t>();

		//parse the cpu part
		cpu_generated_chunk_size = json[cpu_generated_chunk_size_tag_].get<size_t>();

		//nonce
		const auto& nonce_json = json[nonce_tag_];
		prefix = nonce_json[prefix_tag_].get<std::string>();
		suffix = nonce_json[suffix_tag_].get<std::string>();
	}

private:
	
	std::string max_threads_per_block_tag_{ "maxThreadsPerBlock" };
	std::string nonce_tag_{ "nonce" };
	std::string prefix_tag_{ "prefix" };
	std::string suffix_tag_{ "shaSuffix" };
	std::string cpu_generated_chunk_size_tag_{ "cpuGeneratedChunkSize" };
	std::string threads_chunks_number_tag_{ "threadChunksNumber" };
	std::string cuda_tag_{ "cuda" };
	std::string device_name_tag_{ "deviceName" };

};

