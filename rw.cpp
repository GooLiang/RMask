#include <iostream>
#include <vector>
#include <set>
#include <random>
#include <cstdlib>
#include <vector>
#include <random>
extern "C" {
    void set_seed(unsigned int seed) {
        std::srand(seed);
    }
    int*** random_walk_interface(float** adj_matrix, int num_steps, int num_nodes, int* index_ptr) {
        std::vector<std::vector<int>> neighbors(num_nodes);
        neighbors.resize(num_nodes);
        int*** adj_matrix_rw_buffer = new int**[num_steps];
        std::vector<std::vector<int>> row_indices;
        std::vector<std::vector<int>> col_indices;
        for (int node = 0; node < num_nodes; node++) {
            for (int neighbor = 0; neighbor < index_ptr[node]; neighbor++) {
                    neighbors[node].push_back(adj_matrix[node][neighbor]);
            }
        }
        for (int step = 1; step <= num_steps; step++) {
            std::vector<int> row_records;
            std::vector<int> col_records;
            std::vector<std::set<int> > visited(num_nodes, std::set<int>());
            
            for (int node = 0; node < num_nodes; node++) {
                int current_node = node;
                bool flag = true;
                
                for (int current_step = 0; current_step < step; current_step++) {
                    std::vector<int> current_neighbors = neighbors[current_node];
                    std::vector<int> valid_neighbors;

                    for (int n : current_neighbors) {
                            valid_neighbors.push_back(n);
                        }
                    if (valid_neighbors.empty()) {
                        flag = false;
                        break;
                    }
                    int next_node_idx = rand() % valid_neighbors.size();
                    int next_node = valid_neighbors[next_node_idx];
                    current_node = next_node;
                }

                if (flag) {
                    row_records.push_back(node);
                    col_records.push_back(current_node);
                }
            }
            row_indices.push_back(row_records);
            col_indices.push_back(col_records);
        }
        // Convert node_indices and neighbor_indices to int* form
        for (int step = 0; step < num_steps; step++) {
            adj_matrix_rw_buffer[step] = new int*[3];
            adj_matrix_rw_buffer[step][0] = new int[row_indices[step].size()];
            adj_matrix_rw_buffer[step][1] = new int[row_indices[step].size()];
            adj_matrix_rw_buffer[step][2] = new int[1];
            // std::cout << "row_indices[step]" << " ";
            // std::cout << row_indices[step].size() << std::endl;
            for (int i = 0; i < row_indices[step].size(); i++) {
                adj_matrix_rw_buffer[step][0][i] = row_indices[step][i];
                adj_matrix_rw_buffer[step][1][i] = col_indices[step][i];
            }
            adj_matrix_rw_buffer[step][2][0] = row_indices[step].size();
        }
        // Free memory
        return adj_matrix_rw_buffer;
    }
}