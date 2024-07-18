#include "processing_cvsa/Test_CVSA.hpp"

namespace processing{

Test_CVSA::Test_CVSA(void) : nh_("~") { 
    this->pub_ = this->nh_.advertise<processing_cvsa::features>("/cvsa/features", 1);
    this->sub_ = this->nh_.subscribe("/neurodata", 1, &processing::Test_CVSA::on_received_data, this);

    this->buffer_ = new rosneuro::RingBuffer<float>();
    this->has_new_data_ = false;
}

Test_CVSA::~Test_CVSA(){
    this->outputFile_buffer_.close();
    for(int i = 0; i < this->outputFile_filtered_.size(); i++){
        this->outputFile_filtered_[i].close();
    }
    this->outputFile_features_.close();
}

bool Test_CVSA::str2vecOfvec(std::string current_str,  std::vector<std::vector<float>>& out){
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


bool Test_CVSA::configure(void){

    if(ros::param::get("~nchannels", this->nchannels_) == false){
        ROS_ERROR("[Processing] Missing 'nchannels' parameter, which is a mandatory parameter");
        return false;
    }
    if(ros::param::get("~nsamples", this->nsamples_) == false){
        ROS_ERROR("[Processing] Missing 'nsamples' parameter, which is a mandatory parameter");
        return false;
    }

    // Buffer configuration -> TODO: create the yaml to load with buffer parameters
    if(!this->buffer_->configure("RingBufferCfg")){
        ROS_ERROR("[%s] Buffer not configured correctly", this->buffer_->name().c_str());
        return false;
    }

    // Filters parameters
    int filterOrder;
    float sampleRate, avg, windowSize;
    std::string band_str;
    if(ros::param::get("~filter_order", filterOrder) == false){
        ROS_ERROR("[Processing] Missing 'filter_order' parameter, which is a mandatory parameter");
        return false;
    }
    if(ros::param::get("~samplerate", sampleRate) == false){
        ROS_ERROR("[Processing] Missing 'sample_rate' parameter, which is a mandatory parameter");
        return false;
    }
    if(ros::param::get("~filters_band", band_str) == false){
        ROS_ERROR("[Processing] Missing 'filters_band' parameter, which is a mandatory parameter");
        return false;
    }
    if(!this->str2vecOfvec(band_str, this->filters_band_)){
        ROS_ERROR("[Processing] Error in 'filters_band' parameter");
        return false;
    }
    if(ros::param::get("~avg", avg) == false){
        ROS_ERROR("[Processing] Missing 'avg' parameter, which is a mandatory parameter");
        return false;
    }

    // filters parameters
    windowSize = avg *  sampleRate;
    Eigen::VectorXd tmp_b = Eigen::VectorXd::Ones(windowSize) / windowSize;
    std::vector<double> b(tmp_b.data(), tmp_b.data() + tmp_b.size());
    std::vector<double> a = {1.0};

    // Filter configuration
    for(int i = 0; i < this->filters_band_.size(); i++){
        std::cout << "Filtering band: " << this->filters_band_[i][0] << " - " << this->filters_band_[i][1] << std::endl;
        this->filters_low_.push_back(rosneuro::Butterworth<double>(rosneuro::ButterType::LowPass,  filterOrder,  this->filters_band_[i][1], sampleRate));
        this->filters_high_.push_back(rosneuro::Butterworth<double>(rosneuro::ButterType::HighPass,  filterOrder,  this->filters_band_[i][0], sampleRate));
    }

    std::string datapath = "/home/paolo/cvsa_ws/src/processing_cvsa";
    const std::string fileoutput_filtered  = datapath + "/test/test_filtered.csv";
    const std::string fileoutput_buffer  = datapath + "/test/test_buffer.csv"; 
    const std::string fileoutput_features  = datapath + "/test/test_features.csv"; 

    this->outputFile_features_.open(fileoutput_features);
    if(this->outputFile_features_.is_open()){
        std::cout << "file for features opened" << std::endl;
    }
    this->outputFile_buffer_.open(fileoutput_buffer);
    if(this->outputFile_buffer_.is_open()){
        std::cout << "file for buffer opened" << std::endl;
    }
    for(int i = 0; i < this->filters_band_.size(); i++){
        std::string fileoutput_filtered = datapath + "/test/test_filtered_" + std::to_string(i) + ".csv";
        this->outputFile_filtered_.push_back(std::ofstream(fileoutput_filtered));
        if(this->outputFile_filtered_[i].is_open()){
            std::cout << "file for filter " << std::to_string(i) << "opened" << std::endl;
        }
    }

    //this->filter_low_test_ = rosneuro::Butterworth<double>(rosneuro::ButterType::LowPass,  4,  12, 512);
    //this->filter_high_test_ = rosneuro::Butterworth<double>(rosneuro::ButterType::HighPass,  4,  10, 512);

    return true;
}

void Test_CVSA::run(){
    ros::Rate r(512);

    while(ros::ok()){
        if(this->has_new_data_){
            Test_CVSA::ClassifyResults res = this->classify();
            this->has_new_data_ = false;

            if(res == Test_CVSA::ClassifyResults::Error){
                ROS_ERROR("Error in CVSA processing");
                break;
            }else if(res == Test_CVSA::ClassifyResults::BufferNotFull){
                ROS_WARN("Buffer not full");
                continue;
            }

            std::cout << "publishing..." << std::endl;
            this->pub_.publish(this->out_);
        }
        ros::spinOnce();
        r.sleep();
    }
}

void Test_CVSA::on_received_data(const rosneuro_msgs::NeuroFrame &msg){
    this->has_new_data_ = true;

    float* ptr_in;
    float* ptr_eog;
    ptr_in = const_cast<float*>(msg.eeg.data.data());
    ptr_eog = const_cast<float*>(msg.exg.data.data());
    
    this->data_in_ = Eigen::Map<rosneuro::DynamicMatrix<float>>(ptr_in, this->nchannels_, this->nsamples_);
    
}

void Test_CVSA::set_message(std::vector<Eigen::Matrix<double, 1, Eigen::Dynamic>> data, std::vector<std::vector<float>> filters_band){
    // flatten the data
    uint32_t rows = data.size();
    uint32_t cols = data[0].size();

    std::vector<double> c_data;
    data.reserve(rows * cols);
    for (const auto& row : data) {
        c_data.insert(c_data.end(), row.data(), row.data() + row.size());
    }

    std::vector<float> c_bands;
    for(const auto& band : filters_band){
        c_bands.insert(c_bands.end(), band.begin(), band.end());
    }

    this->out_.header.stamp = ros::Time::now();
	this->out_.data = c_data;
    this->out_.cols = cols;
    this->out_.rows = rows;
    this->out_.bands = c_bands;
}

void Test_CVSA::save_features(std::vector<Eigen::Matrix<double, 1, Eigen::Dynamic>> data){
    const static Eigen::IOFormat format(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");

    // flatten the data
    uint32_t rows = data.size();
    uint32_t cols = data[0].size();
    
    std::vector<double> c_data;
    data.reserve(rows * cols);
    for (const auto& row : data) {
        c_data.insert(c_data.end(), row.data(), row.data() + row.size());
    }

    // save the flatten data
    Eigen::RowVectorXd features = Eigen::Map<Eigen::RowVectorXd>(c_data.data(), c_data.size()); 
    this->outputFile_features_ << features.format(format) << std::endl;
}

void Test_CVSA::save_data_filtered(Eigen::MatrixXd data, int idx){
    const static Eigen::IOFormat format(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
    this->outputFile_filtered_[idx] << data.col(0).format(format) << std::endl;
}

void Test_CVSA::save_data_buffer(Eigen::MatrixXd data){
    const static Eigen::IOFormat format(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
    this->outputFile_buffer_ << data.col(0).format(format) << std::endl;
}

Test_CVSA::ClassifyResults Test_CVSA::classify(void){

    this->buffer_->add(this->data_in_.transpose().cast<float>()); // [samples x channels]
    if(!this->buffer_->isfull()){
        return Test_CVSA::ClassifyResults::BufferNotFull;
    }

    Eigen::MatrixXf data_buffer = this->buffer_->get();

    this->save_data_buffer(data_buffer.cast<double>());

    try{
        
        std::vector<Eigen::Matrix<double, 1, Eigen::Dynamic>> all_processed_signals;
        std::vector<std::vector<float>> bands;

        // iterate over all filters
        for(int i = 0; i < this->filters_low_.size(); i++){
            Eigen::MatrixXd data1, data2;
            Eigen::Matrix<double, 1, Eigen::Dynamic> final_data;
            //data2 = this->filter_low_test_.apply(data_buffer.cast<double>());
            //data1 = this->filter_high_test_.apply(data2);
            data2 = this->filters_low_[i].apply(data_buffer.cast<double>());
            data1 = this->filters_high_[i].apply(data2);
            this->save_data_filtered(data1, i);

            data2 = data1.array().pow(2);

            data1 = data2.colwise().mean();

            final_data = data1.array().log();

            all_processed_signals.push_back(final_data);

            bands.push_back({this->filters_band_[i][0], this->filters_band_[i][1]});
        }

        this->save_features(all_processed_signals);

        // send all the data to the classifier (nbands x nchannels)
        this->set_message(all_processed_signals, bands);

        return Test_CVSA::ClassifyResults::Success;

    }catch(std::exception& e){
        ROS_ERROR("Error in CVSA processing: %s", e.what());
        return Test_CVSA::ClassifyResults::Error;
    }
}
}