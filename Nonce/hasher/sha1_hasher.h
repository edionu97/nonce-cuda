#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>


#include "cuda_runtime.h"

#include "../helpers/sha1_helpers.h"
#include "../helpers/string_helpers.h"

class sha1_hasher
{
	/**
	 * \brief
	 * This represents the core of the algorithm, a 512 bit structure used for computing the hash values
	 */
	struct secure_hash_algorithm_context
	{
		uint32_t state[5]{};
		uint32_t count[2]{};
		unsigned char buffer[64]{};

		/**
		 * \brief
		 * This method it is used for creating a copy of the sha instance
		 * \param value_to_be_copied : the value that will be copied
		 * \return  a new instance of the secure_hash_algorithm_context
		 */
		__device__ static secure_hash_algorithm_context get_copy(const secure_hash_algorithm_context& value_to_be_copied) noexcept
		{
			//initialize the context
			secure_hash_algorithm_context ctx{};

			//copy the values
			memcpy(&ctx.state, &value_to_be_copied.state, sizeof(state));
			memcpy(&ctx.count, &value_to_be_copied.count, sizeof(count));
			memcpy(&ctx.buffer, &value_to_be_copied.buffer, sizeof(buffer));

			//return the value
			return ctx;
		}
	};

	/**
	 * \brief
	 * This method it is used for instantiating a new sha context
	 * \return an instance of sha context
	 */
	__device__ static secure_hash_algorithm_context create_sha1_context()
	{
		secure_hash_algorithm_context context{};

		//initialize the states 
		context.state[0] = 0x67452301;
		context.state[1] = 0xEFCDAB89;
		context.state[2] = 0x98BADCFE;
		context.state[3] = 0x10325476;
		context.state[4] = 0xC3D2E1F0;

		//set the values to the count buffer
		context.count[0] = context.count[1] = 0;

		//return the context
		return context;
	}

	/**
	 * \brief
	 * Places the message digest in digest array, which must have space for SHA_DIGEST_LENGTH == 20 bytes of output, and erases the SHA_CTX .
	 * \param digest : the digest array, that contains the 20 bytes of data
	 * \param context : the sha1 context
	 */
	__device__ static void sha1_final(unsigned char digest[20], secure_hash_algorithm_context& context)
	{
		unsigned char final_count[8] = {};
		for (unsigned i = 0; i < 8; i++)
		{
			final_count[i] = static_cast<unsigned char>((context.count[(i >= 4 ? 0 : 1)] >> ((3 - (i & 3)) * 8)) & 255);
		}

		unsigned char character = 0200;
		sha1_update(context, &character, 1);

		while ((context.count[0] & 504) != 448)
		{
			character = 0000;
			sha1_update(context, &character, 1);
		}

		sha1_update(context, final_count, 8);
		for (unsigned i = 0; i < 20; i++)
		{
			digest[i] = static_cast<unsigned char>((context.state[i >> 2] >> ((3 - (i & 3)) * 8)) & 255);
		}

		//clear all the data
		memset(&context, '\0', sizeof(context));
		memset(&final_count, '\0', sizeof(final_count));
	}

	/**
	 * \brief
	 * This method updates the sha1 value (uses the previous computed value to
	 * \param context : represents the sha1 context
	 * \param data : represents the data that will be processed
	 * \param data_length : represents the length of the data
	 */
	__device__ static void sha1_update(
		secure_hash_algorithm_context& context,
		const unsigned char* data,
		const uint32_t data_length)
	{
		auto j = context.count[0];

		if ((context.count[0] += data_length << 3) < j)
		{
			context.count[1]++;
		}

		context.count[1] += (data_length >> 29);

		j = (j >> 3) & 63;

		uint32_t i;
		if ((j + data_length) > 63)
		{
			memcpy(&context.buffer[j], data, (i = 64 - j));

			sha1_transform(context.state, context.buffer);
			for (; i + 63 < data_length; i += 64)
			{
				sha1_transform(context.state, &data[i]);
			}

			j = 0;
		}
		else
		{
			i = 0;
		}

		memcpy(&context.buffer[j], &data[i], data_length - i);
	}

	/**
	 * \brief
	 * This method represents the heart of the algorithm
	 * \param state : the state array
	 * \param buffer : the 64 byte array
	 */
	__device__ static void sha1_transform(uint32_t state[5], const unsigned char buffer[64])
	{
		//create the union
		using char64_long16 = union
		{
			unsigned char c[64];
			uint32_t l[16];
		};

		char64_long16 block[1];
		memcpy(block, buffer, 64);

		//copy in each variable it's corresponding value
		uint32_t a, b, c, d, e;
		a = state[0];
		b = state[1];
		c = state[2];
		d = state[3];
		e = state[4];

		//execute sha1 round 0
		SHA1_R0(a, b, c, d, e, 0);
		SHA1_R0(e, a, b, c, d, 1);
		SHA1_R0(d, e, a, b, c, 2);
		SHA1_R0(c, d, e, a, b, 3);
		SHA1_R0(b, c, d, e, a, 4);
		SHA1_R0(a, b, c, d, e, 5);
		SHA1_R0(e, a, b, c, d, 6);
		SHA1_R0(d, e, a, b, c, 7);
		SHA1_R0(c, d, e, a, b, 8);
		SHA1_R0(b, c, d, e, a, 9);
		SHA1_R0(a, b, c, d, e, 10);
		SHA1_R0(e, a, b, c, d, 11);
		SHA1_R0(d, e, a, b, c, 12);
		SHA1_R0(c, d, e, a, b, 13);
		SHA1_R0(b, c, d, e, a, 14);
		SHA1_R0(a, b, c, d, e, 15);

		//execute sha1 round 1
		SHA1_R1(e, a, b, c, d, 16);
		SHA1_R1(d, e, a, b, c, 17);
		SHA1_R1(c, d, e, a, b, 18);
		SHA1_R1(b, c, d, e, a, 19);

		//execute sha1 round 2
		SHA1_R2(a, b, c, d, e, 20);
		SHA1_R2(e, a, b, c, d, 21);
		SHA1_R2(d, e, a, b, c, 22);
		SHA1_R2(c, d, e, a, b, 23);
		SHA1_R2(b, c, d, e, a, 24);
		SHA1_R2(a, b, c, d, e, 25);
		SHA1_R2(e, a, b, c, d, 26);
		SHA1_R2(d, e, a, b, c, 27);
		SHA1_R2(c, d, e, a, b, 28);
		SHA1_R2(b, c, d, e, a, 29);
		SHA1_R2(a, b, c, d, e, 30);
		SHA1_R2(e, a, b, c, d, 31);
		SHA1_R2(d, e, a, b, c, 32);
		SHA1_R2(c, d, e, a, b, 33);
		SHA1_R2(b, c, d, e, a, 34);
		SHA1_R2(a, b, c, d, e, 35);
		SHA1_R2(e, a, b, c, d, 36);
		SHA1_R2(d, e, a, b, c, 37);
		SHA1_R2(c, d, e, a, b, 38);
		SHA1_R2(b, c, d, e, a, 39);

		//execute sha1 round 3
		SHA1_R3(a, b, c, d, e, 40);
		SHA1_R3(e, a, b, c, d, 41);
		SHA1_R3(d, e, a, b, c, 42);
		SHA1_R3(c, d, e, a, b, 43);
		SHA1_R3(b, c, d, e, a, 44);
		SHA1_R3(a, b, c, d, e, 45);
		SHA1_R3(e, a, b, c, d, 46);
		SHA1_R3(d, e, a, b, c, 47);
		SHA1_R3(c, d, e, a, b, 48);
		SHA1_R3(b, c, d, e, a, 49);
		SHA1_R3(a, b, c, d, e, 50);
		SHA1_R3(e, a, b, c, d, 51);
		SHA1_R3(d, e, a, b, c, 52);
		SHA1_R3(c, d, e, a, b, 53);
		SHA1_R3(b, c, d, e, a, 54);
		SHA1_R3(a, b, c, d, e, 55);
		SHA1_R3(e, a, b, c, d, 56);
		SHA1_R3(d, e, a, b, c, 57);
		SHA1_R3(c, d, e, a, b, 58);
		SHA1_R3(b, c, d, e, a, 59);

		//execute sha1 round 4
		SHA1_R4(a, b, c, d, e, 60);
		SHA1_R4(e, a, b, c, d, 61);
		SHA1_R4(d, e, a, b, c, 62);
		SHA1_R4(c, d, e, a, b, 63);
		SHA1_R4(b, c, d, e, a, 64);
		SHA1_R4(a, b, c, d, e, 65);
		SHA1_R4(e, a, b, c, d, 66);
		SHA1_R4(d, e, a, b, c, 67);
		SHA1_R4(c, d, e, a, b, 68);
		SHA1_R4(b, c, d, e, a, 69);
		SHA1_R4(a, b, c, d, e, 70);
		SHA1_R4(e, a, b, c, d, 71);
		SHA1_R4(d, e, a, b, c, 72);
		SHA1_R4(c, d, e, a, b, 73);
		SHA1_R4(b, c, d, e, a, 74);
		SHA1_R4(a, b, c, d, e, 75);
		SHA1_R4(e, a, b, c, d, 76);
		SHA1_R4(d, e, a, b, c, 77);
		SHA1_R4(c, d, e, a, b, 78);
		SHA1_R4(b, c, d, e, a, 79);

		//add the values back into the state
		state[0] += a;
		state[1] += b;
		state[2] += c;
		state[3] += d;
		state[4] += e;

		//clear the data
		a = b = c = d = e = 0;
		memset(block, '\0', sizeof(block));
	}

	secure_hash_algorithm_context context_;
	bool is_context_initialized_ = false;

public:

	/**
	 * \brief
	 * This method it is used for adding characters/strings into the current computation
	 * \param data : the data that will be added
	 */
	__device__ void update(const char* data)
	{
		if (!is_context_initialized_)
		{
			is_context_initialized_ = true;
			context_ = create_sha1_context();
		}

		//execute the sha1 update logic
		sha1_update(
			context_,
			reinterpret_cast<unsigned char*>(const_cast<char*>(data)),
			static_cast<uint32_t>(string_helpers::device_strlen(data)));
	}

	/**
	 * \brief
	 * This method get the final digest figures, and clears all the data
	 */
	__device__ void get_final(char* result_buffer)
	{
		//get the results
		uint8_t results[20];
		sha1_final(results, context_);

		//iterate through characters
		char hex_buffer[3] = {};
		for (auto character : results)
		{
			//convert the int value to hex values
			memset(hex_buffer, 0, sizeof(hex_buffer));

			//convert the character into hex value
			// ReSharper disable once CppDeprecatedEntity
			//sprintf(hex_buffer, "%02x", character);
			string_helpers::byte_to_hex(character, hex_buffer);

			//copy the result_buffer into the final buffer
			memcpy(result_buffer + string_helpers::device_strlen(result_buffer), hex_buffer, string_helpers::device_strlen(hex_buffer));
		}

		is_context_initialized_ = false;
	}

	/**
	 * \brief
	 * Getter for context (creates a copy of context and returns it)
	 * \return a reference to sha1 context
	 */
	__device__ secure_hash_algorithm_context get_context_copy() const 
	{
		return secure_hash_algorithm_context::get_copy(context_);
	}
	
	/**
	 * \brief
	 * Context setter
	 * \param new_context : the new value for the context
	 */
	__device__ void set_context(const secure_hash_algorithm_context& new_context)
	{
		context_ = new_context;
		is_context_initialized_ = true;
	}
};

