#pragma once
#include <cstddef>
#include "device_launch_parameters.h"

class string_helpers
{
public:


	/**
	 * \brief
	 *	This function it is used for computing the size of a string
	 * \param array : the array that
	 * \return a number representing the total number of characters
	 */
	__device__ static size_t device_strlen(const char* array)
	{
		size_t index = 0;

		//as long as we do not encounter a null value, we advance
		while (*(array + index) != NULL)
		{
			++index;
		}

		return index;
	}

	
	/**
	 * \brief
	 * This function it is used for converting a number (0-255) from decimal into hex 
	 * \param number : the number that will be converted
	 * \param representation 
	 */
	__device__ static void byte_to_hex(uint8_t number, char* representation)
	{
		//initially the number is 00
		representation[0] = representation[1] = static_cast<char>(48);
		
		//convert the number to binary
		auto no_elements = 1;
		for (; number != 0; number /= 16, --no_elements)
		{
			const auto reminder = number % 16;
			if (reminder < 10)
			{
				representation[no_elements] = static_cast<char>(48 + reminder);
				continue;
			}

			representation[no_elements] = static_cast<char>(87 + reminder);
		}
	}
};

