// SPDX-License-Identifier: MPL-2.0
#pragma once
#include "aare/ClusterFile.hpp"
#include "aare/ClusterVector.hpp"
#include "aare/Dtype.hpp"
#include "aare/NDArray.hpp"
#include "aare/NDView.hpp"
#include "aare/Pedestal.hpp"
#include "aare/defs.hpp"
#include <cstddef>

namespace aare {

template <typename ClusterType,
          typename = std::enable_if_t<is_cluster_v<ClusterType>>>
struct no_2x2_cluster {
    constexpr static bool value =
        ClusterType::cluster_size_x > 2 && ClusterType::cluster_size_y > 2;
};

template <typename ClusterType = Cluster<int32_t, 3, 3>,
          typename FRAME_TYPE = uint16_t, typename PEDESTAL_TYPE = double,
          typename = std::enable_if_t<no_2x2_cluster<ClusterType>::value>>
class ClusterFinder {
    Shape<2> m_image_size;
    PEDESTAL_TYPE m_nSigma;
    const PEDESTAL_TYPE c2;
    const PEDESTAL_TYPE c3;

    Pedestal<PEDESTAL_TYPE> m_pedestal;
    ClusterVector<ClusterType> m_clusters;

    static constexpr uint8_t ClusterSizeX = ClusterType::cluster_size_x;
    static constexpr uint8_t ClusterSizeY = ClusterType::cluster_size_y;
    static constexpr int dx = ClusterSizeX / 2;
    static constexpr int dy = ClusterSizeY / 2;
    using CT = typename ClusterType::value_type;

  public:
    /**
     * @brief Construct a new ClusterFinder object
     * @param image_size size of the image
     * @param cluster_size size of the cluster (x, y)
     * @param nSigma number of sigma above the pedestal to consider a photon
     * @param capacity initial capacity of the cluster vector
     *
     */
    ClusterFinder(Shape<2> image_size, PEDESTAL_TYPE nSigma = 5.0,
                  size_t capacity = 1000000)
        : m_image_size(image_size), m_nSigma(nSigma),
          c2(sqrt((ClusterSizeY + 1) / 2 * (ClusterSizeX + 1) / 2)),
          c3(sqrt(ClusterSizeX * ClusterSizeY)),
          m_pedestal(image_size[0], image_size[1]), m_clusters(capacity) {
        LOG(logDEBUG) << "ClusterFinder: "
                      << "image_size: " << image_size[0] << "x" << image_size[1]
                      << ", nSigma: " << nSigma << ", capacity: " << capacity;
    }

    void set_nSigma(PEDESTAL_TYPE nSigma) { m_nSigma = nSigma; }

    PEDESTAL_TYPE get_nSigma() const { return m_nSigma; }

    void push_pedestal_frame(NDView<FRAME_TYPE, 2> frame) {
        m_pedestal.push(frame);
    }

    NDArray<PEDESTAL_TYPE, 2> pedestal() { return m_pedestal.mean(); }
    NDArray<PEDESTAL_TYPE, 2> noise() { return m_pedestal.std(); }
    void clear_pedestal() { m_pedestal.clear(); }

    /**
     * @brief Move the clusters from the ClusterVector in the ClusterFinder to a
     * new ClusterVector and return it.
     * @param realloc_same_capacity if true the new ClusterVector will have the
     * same capacity as the old one
     *
     */
    ClusterVector<ClusterType>
    steal_clusters(bool realloc_same_capacity = false) {
        ClusterVector<ClusterType> tmp = std::move(m_clusters);
        if (realloc_same_capacity)
            m_clusters = ClusterVector<ClusterType>(tmp.capacity());
        else
            m_clusters = ClusterVector<ClusterType>{};
        return tmp;
    }
    void find_clusters(NDView<FRAME_TYPE, 2> frame, uint64_t frame_number = 0,
                       bool update_pedestal = true) {
        m_clusters.set_frame_number(frame_number);
        for (int iy = dy; iy < frame.shape(0) - dy; iy++) {
            for (int ix = dx; ix < frame.shape(1) - dx; ix++) {
                PEDESTAL_TYPE rms = m_pedestal.std(iy, ix);
                PEDESTAL_TYPE value = (frame(iy, ix) - m_pedestal.mean(iy, ix));

                // negative value below pedestal threshold => skip
                if (value < -m_nSigma * rms)
                    continue;

                std::array<PEDESTAL_TYPE, ClusterSizeX * ClusterSizeY>
                    cluster_values;
                PEDESTAL_TYPE total = 0;
                PEDESTAL_TYPE max = std::numeric_limits<FRAME_TYPE>::min();

                for (int ir = -dy; ir < dy + 1; ir++) {
                    for (int ic = -dx; ic < dx + 1; ic++) {
                        PEDESTAL_TYPE val = frame(iy + ir, ix + ic) -
                                            m_pedestal.mean(iy + ir, ix + ic);
                        cluster_values[(ir + dy) * ClusterSizeX + (ic + dx)] =
                            val;
                        total += val;
                        max = std::max(max, val);
                    }
                }

                if (max > m_nSigma * rms || total > c3 * m_nSigma * rms) {
                    if (value == max) {
                        ClusterType cluster{};
                        cluster.x = ix;
                        cluster.y = iy;
                        for (size_t i = 0; i < ClusterSizeX * ClusterSizeY;
                             i++) {
                            // If the cluster type is an integral type, and the
                            // pedestal is a floating point type then we need to
                            // round the value before storing it
                            if constexpr (std::is_integral_v<CT> &&
                                          std::is_floating_point_v<
                                              PEDESTAL_TYPE>) {
                                auto tmp = std::lround(cluster_values[i]);
                                cluster.data[i] = static_cast<CT>(tmp);
                            }
                            // On the other hand if both are floating point or
                            // both are integral then we can just static cast
                            // directly
                            else {
                                auto tmp = cluster_values[i];
                                cluster.data[i] = static_cast<CT>(tmp);
                            }
                        }
                        // Add the cluster to the output ClusterVector
                        m_clusters.push_back(cluster);
                    } else {
                        continue;
                    }
                } else {
                    if (update_pedestal) {
                        m_pedestal.push_fast(
                            iy, ix,
                            frame(
                                iy,
                                ix)); // Assume we have reached n_samples in the
                    }
                }
            }
        }
    }
};

} // namespace aare