#pragma once
#include <algorithm>
#include <driver_types.h>
#include <iostream>
#include <numeric>
#include <ostream>
#include <vector>

#include "cuda_device_helpers.h"

namespace cuda_print_utils
{
	/// <summary>
	/// This method it is used for printing at the console the device properties
	/// </summary>
	/// <param name="out">the output stream</param>
	/// <param name="props">the device properties</param>
	/// <returns>the same stream, for operator chaining</returns>
	inline auto operator <<(std::ostream& out, const cudaDeviceProp& props) -> std::ostream&
	{
		//device generic information
		out << "Device name: " << props.name << '\n';
		out << "Dedicated GPU: " << std::boolalpha << !props.integrated << '\n';
		out << "Compute capability: " << props.major << '.' << props.minor << '\n';
		out << "Total memory: " << props.totalGlobalMem / 1020 / 1024 << '\n';

		//threading information
		out << "Max threads dim: " << props.maxThreadsDim[0] << " x ";
		out << props.maxThreadsDim[1] << " x ";
		out << props.maxThreadsDim[2] << '\n';

		out << "Max threads per block: " << props.maxThreadsPerBlock << '\n';
		out << "Warp size:" << props.warpSize << '\n';

		out << "Registries per block: " << props.regsPerBlock << '\n';

		//return instance of the stream
		return  out;
	}

}

namespace statistics
{
	class elapsed_time_computer  // NOLINT(cppcoreguidelines-special-member-functions)
	{
		std::vector<std::pair<cudaEvent_t, std::string>> cuda_events_;
		std::vector<float> elapsed_time_;

	public:

		/// <summary>
		/// This method will start the event and will start recording
		/// </summary>
		/// <param name="event_label">the label of the event, by default is null</param>
		/// <param name="is_last_event"></param>
		void set_time_period(const std::string& event_label = "", const bool is_last_event = false)
		{
			//create the event
			cudaEvent_t event;
			cuda_device_helpers::check(cudaEventCreate(&event));

			//start event recording
			cuda_device_helpers::check(cudaEventRecord(event));

			if (is_last_event)
			{
				cudaEventSynchronize(event);
			}

			//push the events
			cuda_events_.emplace_back(event, event_label);
		}

		/// <summary>
		/// This function it is used for computing the total elapsed time
		/// </summary>
		/// <returns></returns>
		float get_total_time_as_ms() const
		{
			if (cuda_events_.size() <= 1)
			{
				return .0;
			}

			//get the first time period
			const auto& start = cuda_events_[0];
			const auto& end = cuda_events_.back();

			//compute the difference 
			auto elapsed_time = .0f;
			cuda_device_helpers::check(cudaEventElapsedTime(&elapsed_time, start.first, end.first));

			return  elapsed_time;
		}

		/// <summary>
		/// Reset the elapsed time
		/// </summary>
		void reset()
		{
			cuda_events_.clear();
		}

		/// <summary>
		/// This method will get two consecutive events and will compute the elapsed time between them
		/// </summary>
		/// <param name="out">the stream into which we are writing</param>
		void print_time_periods(std::ostream& out = std::cout) const
		{
			for (size_t idx = 1; idx < cuda_events_.size(); ++idx)
			{
				//get the events
				const auto& start = cuda_events_[idx - 1];
				const auto& end = cuda_events_[idx];

				//compute the time
				auto elapsed_time = .0f;
				cuda_device_helpers::check(cudaEventElapsedTime(&elapsed_time, start.first, end.first));

				//print the duration
				out << end.second << elapsed_time / 1000 << " sec\n";
			}

			out << "Total elapsed time: " << get_total_time_as_ms() / 1000 << " sec\n";
		}

		/// <summary>
		/// This method it is used to compute standard deviation and print the latest result 
		/// </summary>
		/// <param name="out">the output stream</param>
		void report(std::ostream& out)
		{
			out << "Computed time periods for last round values\n\n";
			
			//print time periods
			print_time_periods(out);

			if(elapsed_time_.empty())
			{
				return;
			}

			const auto repeat_no = static_cast<float>(elapsed_time_.size());
			const auto avg_value = std::accumulate(elapsed_time_.begin(), elapsed_time_.end(), .0f) / repeat_no;
			const auto min_value = *std::min_element(elapsed_time_.begin(), elapsed_time_.end());
			const auto max_value = *std::max_element(elapsed_time_.begin(), elapsed_time_.end());

			auto standard_deviation = .0f;
			for (size_t r = 0; r < repeat_no; r++) {
				standard_deviation += (elapsed_time_[r] - avg_value) * (elapsed_time_[r] - avg_value);
			}
			standard_deviation = sqrt(standard_deviation / static_cast<float>(repeat_no));

			out << "\n\nAll rounds values\n";
			out << "Min: " << min_value << ' ';
			out << "Max: " << max_value << ' ';
			out << "Mean: " << elapsed_time_[repeat_no / 2] << ' ';
			out << "Avg: " << avg_value << ' ';
			out << "StandardDeviation: " << standard_deviation << '\n';
		}

		/// <summary>
		/// This method it is used for adding an elapsed time
		/// </summary>
		/// <param name="elapsed_time"></param>
		void add_elapsed_time(const float elapsed_time)
		{
			elapsed_time_.push_back(elapsed_time);
		}

		~elapsed_time_computer() noexcept
		{
			for (const auto& cuda_event : cuda_events_)
			{
				cudaEventDestroy(cuda_event.first);
			}
		}
	};


}
