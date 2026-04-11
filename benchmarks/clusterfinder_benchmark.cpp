// SPDX-License-Identifier: MPL-2.0
#include "aare/ClusterFinder.hpp"
#include "aare/File.hpp"
#include "aare/Frame.hpp"
#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>
namespace {

constexpr aare::Shape<2> kImageShape{400, 400};
constexpr std::size_t kPedestalSamples = 1500;
constexpr std::size_t kTestFrames = 5000;

using ClusterType = aare::Cluster<int32_t, 3, 3>;
using Finder = aare::ClusterFinder<ClusterType, uint16_t, double>;

class BenchmarkData {
  public:
    std::vector<aare::Frame> v_pedestal_frames, v_test_frames;
    size_t n_pedestal_frames, n_test_frames;
    BenchmarkData() {
        auto pedestal_filepath = std::getenv("PEDESTAL_FILE");
        auto data_filepath = std::getenv("DATA_FILE");
        auto pedestal_frames = std::getenv("N_PEDESTAL_FRAMES");
        auto test_frames = std::getenv("N_TEST_FRAMES");
        // if not initialized, take 1000 pedestal and 1000 test frames
        n_pedestal_frames =
            pedestal_frames ? std::stoul(pedestal_frames) : 1000;
        n_test_frames = test_frames ? std::stoul(test_frames) : 1000;
        if (!pedestal_filepath || !data_filepath) {
            throw std::runtime_error("PEDESTAL_FILE and DATA_FILE environment "
                                     "variables must be set");
        }
        aare::File pedestal_file(pedestal_filepath);
        aare::File data_file(data_filepath);
        pedestal_file.seek(0);
        try {
            v_pedestal_frames = pedestal_file.read_n(n_pedestal_frames);
        } catch (const std::exception &e) {
            throw std::runtime_error("Error reading pedestal " +
                                     std::to_string(n_pedestal_frames) +
                                     " frames: " + std::string(e.what()));
        }
        data_file.seek(0);
        try {
            v_test_frames = data_file.read_n(n_test_frames);
        } catch (const std::exception &e) {
            throw std::runtime_error("Error reading test " +
                                     std::to_string(n_test_frames) +
                                     " frames: " + std::string(e.what()));
        }
    }

    void initalize_finder(Finder &finder) {
        for (aare::Frame &frame : v_pedestal_frames) {
            finder.push_pedestal_frame(frame.view<uint16_t>());
        }
    }
};

// Added 'UseNewMethod' template parameter
template <typename FinderType, bool UseNewMethod>
void run_benchmark(benchmark::State &state, const char *label,
                   bool update_pedestal) {
    FinderType finder(kImageShape);
    BenchmarkData data{};
    data.initalize_finder(finder);
    std::size_t cluster_count = 0;
    for (auto _ : state) {
        cluster_count = 0;
        for (aare::Frame &frame : data.v_test_frames) {
            // Resolves at compile-time for zero overhead
            if constexpr (UseNewMethod) {
                finder.find_clusters(frame.view<uint16_t>(), 0,
                                     update_pedestal);
            } else {
                finder.find_clusters_old(frame.view<uint16_t>(), 0);
            }
            cluster_count += finder.steal_clusters(true).size();
        }

        benchmark::DoNotOptimize(cluster_count);
    }

    auto n_test_frames = data.n_test_frames;
    state.counters["test_frames"] = n_test_frames;
    state.counters["clusters_per_frame"] =
        static_cast<double>(cluster_count) / static_cast<double>(n_test_frames);
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(n_test_frames));
    state.SetLabel(label);
}

// Wrapper for the old function
void BM_ClusterFinder_Old(benchmark::State &state) {
    run_benchmark<Finder, false>(state, "find_clusters_old", true);
}

// Wrapper for the new function
void BM_ClusterFinder_New(benchmark::State &state) {
    run_benchmark<Finder, true>(state, "find_clusters", true);
}

void BM_ClusterFinder_New_SkipPedestals(benchmark::State &state) {
    run_benchmark<Finder, true>(state, "find_clusters", false);
}
} // namespace

// Register both benchmarks
BENCHMARK(BM_ClusterFinder_Old)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ClusterFinder_New)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ClusterFinder_New_SkipPedestals)->Unit(benchmark::kMillisecond);