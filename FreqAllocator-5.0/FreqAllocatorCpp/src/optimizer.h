#pragma once
#include <vector>
#include <functional>
#include "single_qubit_model.h"

namespace PSO
{
    // Particle class
    class Particle {
    public:
        std::vector<double> position;
        std::vector<double> velocity;
        std::vector<double> best_position;
        double best_loss;
        std::vector<std::pair<double, double>> ranges;

        inline Particle(int dimensions, const std::vector<std::pair<double, double>>& ranges)
            : best_loss(std::numeric_limits<double>::max()), ranges(ranges) {
            position.resize(dimensions);
            velocity.resize(dimensions);
            best_position.resize(dimensions);
            initializePosition();
        }

        inline void initializePosition() {
            auto& gen = RandomEngine::get_instance();
            for (size_t i = 0; i < position.size(); ++i) {
                std::uniform_real_distribution<> dis(ranges[i].first, ranges[i].second);
                position[i] = dis(gen);
            }
        }

        inline void updatePosition() {
            for (size_t i = 0; i < position.size(); ++i) {
                position[i] += velocity[i];
                // Constrain the position within the specified range
                position[i] = std::max(position[i], ranges[i].first);
                position[i] = std::min(position[i], ranges[i].second);
            }
        }

        inline void updateVelocity(const std::vector<double>& global_best_position, double inertia, double cognitive, double social) {
            std::uniform_real_distribution<> dis(0, 1);
            auto& gen = RandomEngine::get_instance();
            for (size_t i = 0; i < position.size(); ++i) {
                velocity[i] = inertia * velocity[i]
                    + cognitive * dis(gen) * (best_position[i] - position[i])
                    + social * dis(gen) * (global_best_position[i] - position[i]);
            }
        }
    };

    // PSO Algorithm
    inline void ParticleSwarmOptimization(std::vector<Particle>& particles, 
        std::function<double(const std::vector<double>&)> loss,
        int max_iter, double inertia, double cognitive, double social) {
        // std::random_device rd;
        // std::mt19937 gen(rd());
        // auto& gen = RandomEngine::get_instance();

        int dimensions = particles[0].position.size();
        std::vector<double> global_best_position(dimensions);
        double global_best_loss = std::numeric_limits<double>::max();

        for (int iter = 0; iter < max_iter; ++iter) {
            for (Particle& p : particles) {
                double current_loss = loss(p.position);
                // fmt::print("current_loss = {}\n", current_loss);
                if (current_loss < p.best_loss) {
                    p.best_loss = current_loss;
                    p.best_position = p.position;
                }

                if (current_loss < global_best_loss) {
                    global_best_loss = current_loss;
                    global_best_position = p.position;
                }
            }

            for (Particle& p : particles) {
                p.updateVelocity(global_best_position, inertia, cognitive, social);
                p.updatePosition();
            }

            std::cout << "Iteration " << iter << " - Best Global Loss: " << global_best_loss << std::endl;
        }
    }


}