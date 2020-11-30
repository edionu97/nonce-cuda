#pragma once
#include <string>
#include <vector>

#include "cuda_runtime.h"

class nonce_helpers
{
public:


	/**
	 * \brief
	 * This method it is used for converting a vector of strings into an array of linearized data
	 * \param word_matrix : the array that represents a matrix
	 * \return an association between the position in which each word ends and the char array
	 */
	__host__ static std::pair<std::vector<size_t>, std::vector<char>> liniearize_word_matrix_array(const std::vector<std::string>& word_matrix)
	{
		std::vector<char> char_array;
		std::vector<size_t> word_ends_at;

		//iterate through all the words
		size_t character_idx = 0;
		for (const auto& word : word_matrix)
		{
			//convert words into characters
			for (const auto& character : word)
			{
				++character_idx;
				char_array.push_back(character);
			}

			word_ends_at.push_back(++character_idx);
			char_array.push_back(NULL);
		}

		//create the pair of items
		return { word_ends_at, char_array };
	}

	/**
	 * \brief
	 * This method it is used for getting the n-th word from a char array
	 * \param n : the index of word
	 * \param word_array : the char word_array
	 * \param word_boundaries : the vector that indicates each word boundary
	 * \return a pointer indicating the start of the n-th word
	 */
	__device__ static const char* get_nth_word(const size_t n, const char* word_array, const size_t* word_boundaries)
	{
		//assume that first word starts on the 0 position from char array
		if (n == 0)
		{
			return word_array;
		}

		//return the next word offset (the word start address)
		return word_array + word_boundaries[n - 1];
	}

	
	/**
	 * \brief
	 * Compute the min value between two numbers
	 * \tparam T : template parameter
	 * \param a : the first number
	 * \param b : the second number
	 * \return a if a < b otherwise b
	 */
	template<typename  T>
	__device__ static T min(const T a, const T b)
	{
		return a < b ? a : b;
	}
};

