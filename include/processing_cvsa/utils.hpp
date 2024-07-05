#include <string>
#include <eigen3/Eigen/Dense>
#include <vector>

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
    unsigned int nrows;
    unsigned int ncols;
    std::vector<std::vector<float>> temp_matrix;

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

    nrows = out.size();
    ncols = out.at(0).size();

    // check if always same dimension in temp_matrix
    for(auto it=out.begin(); it != out.end(); ++it){
        if((*it).size() != ncols){
            return false;
        }
    }

    return true;
}