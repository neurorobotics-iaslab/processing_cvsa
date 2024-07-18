#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <string>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <fstream>

template<typename T>
void writeCSV(const std::string& filename, const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>&  matrix) {
	const static Eigen::IOFormat format(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");

	std::ofstream file(filename);
	if (file.is_open()) {
		file << matrix.format(format);
		file.close();
	}
}

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> readCSV(const std::string& filename) {

	std::vector<T> values;

	std::ifstream file(filename);
	std::string row;
	std::string entry;
	int nrows = 0;

	while (getline(file, row)) {
		std::stringstream rowstream(row);

		while (getline(rowstream, entry, ',')) {
			values.push_back(std::stod(entry));
		}
		nrows++; 
	}

	return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(values.data(), nrows, values.size() / nrows);

}

template<typename T>
bool str2vecOfvec(std::string current_str,  std::vector<std::vector<T>>& out){
    unsigned int ncols;

    std::stringstream ss(current_str);
    std::string c_row;

    while(getline(ss, c_row, ';')){
        std::stringstream iss(c_row);
        float index;
        std::vector<float> row;
        while(iss >> index){
            row.push_back(index);
        }
        out.push_back(row);
    }

    ncols = out.at(0).size();

    // check if always same dimension in temp_matrix
    for(auto it=out.begin(); it != out.end(); ++it){
        if((*it).size() != ncols){
            return false;
        }
    }

    return true;
}

template<typename T>
int idx_from2vec(std::vector<std::vector<T>> all, Eigen::VectorXf vec){
    std::vector<T> stdVec(vec.data(), vec.data() + vec.size());
    for(int i = 0; i < all.size(); i++){
        if(all.at(i) == stdVec){
            return i;
        }
    }
    return -1;
}

template<typename T>
Eigen::Matrix<T, 1, Eigen::Dynamic> get_features(std::vector<Eigen::Matrix<T, 1, Eigen::Dynamic>> in, std::vector<uint32_t> idchans, Eigen::MatrixXf features_bands, std::vector<std::vector<float>> all_band){
    Eigen::Matrix<T, 1, Eigen::Dynamic> out;
    std::vector<T> tmp_out;
    for(int i = 0; i < idchans.size(); i++){
        Eigen::VectorXf c_band = features_bands.row(i);
        int idx_band = idx_from2vec(all_band, c_band);
        if(idx_band == -1){
            ROS_ERROR("Error in the extraction of the features");
            ros::shutdown();
        }

        tmp_out.push_back(in.at(idx_band).col(idchans.at(i)-1)(0));  // c++ start from 0 and not from 1 as matlab
    }

    out = Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>>(tmp_out.data(), 1, tmp_out.size());

    return out;
}

#endif