#pragma once

#include <nlohmann/json.hpp>
#include <fstream>

#include "../config/configuration.h"


class resources_manager
{
public:

	/// <summary>
	/// Read the configuration
	/// </summary>
	/// <returns>a new instance of config</returns>
	static configuration get_config()
	{
		//read the json from file
		nlohmann::json json;
		std::ifstream{ "resources/config.json" } >> json;

		//return the config
		return configuration(json);
	}
};
