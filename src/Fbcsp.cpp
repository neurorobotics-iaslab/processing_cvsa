#include "processing_bci/Fbcsp.hpp"
#include <XmlRpcException.h>

namespace processing {

Fbcsp::Fbcsp(void) : nh_("~") { 
    this->name_ = "FBCSP Processing";
    this->sub_ = this->nh_.subscribe("/neurodata", 1, &processing::Fbcsp::on_received_data, this);

    this->has_new_data_ = false;
    this->is_configured_ = false;
}

Fbcsp::~Fbcsp() {
    for(auto& buf : this->buffers_){
        delete buf;
    }
}

bool Fbcsp::load_matrix(const std::string& param_name, Eigen::MatrixXd& mat) {
    XmlRpc::XmlRpcValue xml_mat;
    if (!this->nh_.getParam(param_name, xml_mat)) {
        return false;
    }
    if (xml_mat.getType() != XmlRpc::XmlRpcValue::TypeArray) return false;
    int rows = xml_mat.size();
    if (rows == 0) return false;
    
    XmlRpc::XmlRpcValue row0 = xml_mat[0];
    if (row0.getType() != XmlRpc::XmlRpcValue::TypeArray) return false;
    int cols = row0.size();
    
    mat.resize(rows, cols);
    for (int i = 0; i < rows; ++i) {
        XmlRpc::XmlRpcValue row = xml_mat[i];
        if (row.getType() != XmlRpc::XmlRpcValue::TypeArray || row.size() != cols) return false;
        for (int j = 0; j < cols; ++j) {
            XmlRpc::XmlRpcValue val = row[j];
            if (val.getType() == XmlRpc::XmlRpcValue::TypeDouble) {
                mat(i, j) = static_cast<double>(val);
            } else if (val.getType() == XmlRpc::XmlRpcValue::TypeInt) {
                mat(i, j) = static_cast<double>(static_cast<int>(val));
            } else {
                return false;
            }
        }
    }
    return true;
}

bool Fbcsp::load_matrices(const std::string& param_name, std::vector<Eigen::MatrixXd>& mats) {
    XmlRpc::XmlRpcValue xml_mats;
    if (!this->nh_.getParam(param_name, xml_mats)) return false;
    if (xml_mats.getType() != XmlRpc::XmlRpcValue::TypeArray) return false;
    
    for (int k = 0; k < xml_mats.size(); ++k) {
        XmlRpc::XmlRpcValue xml_mat = xml_mats[k];
        if (xml_mat.getType() != XmlRpc::XmlRpcValue::TypeArray) return false;
        int rows = xml_mat.size();
        if (rows == 0) return false;
        XmlRpc::XmlRpcValue row0 = xml_mat[0];
        if (row0.getType() != XmlRpc::XmlRpcValue::TypeArray) return false;
        int cols = row0.size();
        
        Eigen::MatrixXd mat(rows, cols);
        for (int i = 0; i < rows; ++i) {
            XmlRpc::XmlRpcValue row = xml_mat[i];
            if (row.getType() != XmlRpc::XmlRpcValue::TypeArray || row.size() != cols) return false;
            for (int j = 0; j < cols; ++j) {
                XmlRpc::XmlRpcValue val = row[j];
                if (val.getType() == XmlRpc::XmlRpcValue::TypeDouble) {
                    mat(i, j) = static_cast<double>(val);
                } else if (val.getType() == XmlRpc::XmlRpcValue::TypeInt) {
                    mat(i, j) = static_cast<double>(static_cast<int>(val));
                } else {
                    return false;
                }
            }
        }
        mats.push_back(mat);
    }
    return true;
}

bool Fbcsp::configure(void) {
    if(ros::param::get("~nchannels", this->nchannels_) == false){
        ROS_ERROR("[%s] Missing 'nchannels' parameter", this->name_.c_str());
        return false;
    }
    if(ros::param::get("~signal_type", this->signal_type_) == false){
        ROS_ERROR("[%s] Missing 'signal_type' parameter", this->name_.c_str());
        return false;
    }
    if(ros::param::get("~chunkSize", this->chunkSize_) == false){
        ROS_ERROR("[%s] Missing 'chunkSize' parameter", this->name_.c_str());
        return false;
    }
    if(ros::param::get("~run_mode", this->run_mode_) == false){
        ROS_ERROR("[%s] Missing 'run_mode' parameter", this->name_.c_str());
        return false;
    }

    std::string topic_to_pub;
    this->nh_.param<std::string>("topic_to_pub", topic_to_pub, "/eeg_fbcsp");
    ROS_INFO("[%s] topic_to_pub set to: %s", this->name_.c_str(), topic_to_pub.c_str());
    this->pub_ = this->nh_.advertise<processing_bci::eeg_fbcsp>(topic_to_pub, 1);

    // ICA Matrix
    this->nh_.param<bool>("do_ica", this->do_ica_, true);
    if (this->do_ica_) {
        if (!this->load_matrix("ica_matrix", this->ica_matrix_)) {
            ROS_ERROR("[%s] Failed to load 'ica_matrix' from parameter server", this->name_.c_str());
            return false;
        }
        ROS_INFO("[%s] Loaded ICA matrix of size %ldx%ld", this->name_.c_str(), this->ica_matrix_.rows(), this->ica_matrix_.cols());
    }

    // CAR
    this->nh_.param<bool>("do_car", this->do_car_, true);
    if(this->do_car_){
        ROS_INFO("[%s] CAR filter enable", this->name_.c_str());
        this->car_filter_ = rosneuro::Car<double>();
        this->car_filter_.configure("CarCfg");
    }

    // Filters parameters
    int filterOrder;
    float sampleRate;
    std::string band_str;
    if(ros::param::get("~filter_order", filterOrder) == false){
        ROS_ERROR("[%s] Missing 'filter_order' parameter", this->name_.c_str());
        return false;
    }
    if(ros::param::get("~samplerate", sampleRate) == false){
        ROS_ERROR("[%s] Missing 'samplerate' parameter", this->name_.c_str());
        return false;
    }
    if(ros::param::get("~filters_band", band_str) == false){
        ROS_ERROR("[%s] Missing 'filters_band' parameter", this->name_.c_str());
        return false;
    }
    if(!str2vecOfvec<float>(band_str, this->filters_band_)){
        ROS_ERROR("[%s] Error in 'filters_band' parameter", this->name_.c_str());
        return false;
    }
    this->nfilters_ = this->filters_band_.size();

    // CSP Matrices
    if (!this->load_matrices("csp_matrices", this->csp_matrices_)) {
        ROS_ERROR("[%s] Failed to load 'csp_matrices' from parameter server", this->name_.c_str());
        return false;
    }
    if (this->csp_matrices_.size() != this->nfilters_) {
        ROS_ERROR("[%s] Number of CSP matrices (%ld) does not match number of frequency bands (%d)", 
                  this->name_.c_str(), this->csp_matrices_.size(), this->nfilters_);
        return false;
    }
    this->ncomponents_ = this->csp_matrices_[0].rows();
    ROS_INFO("[%s] Loaded %ld CSP matrices, %d components per band", this->name_.c_str(), this->csp_matrices_.size(), this->ncomponents_);

    // Filter configuration
    for(int i = 0; i < this->nfilters_; i++){
        this->filters_low_.push_back(rosneuro::Butterworth<double>(rosneuro::ButterType::LowPass,  filterOrder,  this->filters_band_[i][1], sampleRate));
        this->filters_high_.push_back(rosneuro::Butterworth<double>(rosneuro::ButterType::HighPass,  filterOrder,  this->filters_band_[i][0], sampleRate));
    }

    // Buffer configuration 
    for(int i = 0; i < this->nfilters_; i++){
        this->buffers_.push_back(new rosneuro::RingBuffer<float>());
        if(!this->buffers_.back()->configure("RingBufferCfg")){
            ROS_ERROR("[%s %.2f-%.2f Hz] Buffer not configured correctly", 
                this->buffers_.back()->name().c_str(),
                this->filters_band_[i][0], 
                this->filters_band_[i][1]);
            return false;
        }
    }

    this->is_configured_ = true;
    return true;
}

void Fbcsp::run() {
    ros::Rate r(512);
    if(this->is_configured_ == false){
        ROS_ERROR("[%s] Fbcsp not configured correctly", this->name_.c_str());
        return;
    }

    while(ros::ok()){
        if(this->has_new_data_){
            Fbcsp::ApplyResults res = this->apply();
            this->has_new_data_ = false;
            
            if(res == Fbcsp::ApplyResults::Error){
                ROS_ERROR("[%s] Error in FBCSP processing", this->name_.c_str());
                break;
            }else if(res == Fbcsp::ApplyResults::BufferNotFull){
                ROS_WARN("[%s] Buffer not full", this->name_.c_str());
                this->set_message(Eigen::MatrixXd::Ones(this->ncomponents_, this->nfilters_)); 
            }

            this->pub_.publish(this->out_);
        }
        ros::spinOnce();
        r.sleep();
    }
}

void Fbcsp::on_received_data(const rosneuro_msgs::NeuroFrame &msg) {
    this->has_new_data_ = true;

    float* ptr_in = const_cast<float*>(msg.eeg.data.data());
    float* ptr_eog = const_cast<float*>(msg.exg.data.data());
    
    if(this->run_mode_ == "online"  || (this->run_mode_ == "offline" && this->signal_type_ == "eeg")){
        this->data_in_ = Eigen::Map<rosneuro::DynamicMatrix<float>>(ptr_in, this->nchannels_, this->chunkSize_); // channels x samples
    }else if(this->run_mode_ == "offline" && this->signal_type_ == "eeg_eog"){
        Eigen::MatrixXf eeg_data = Eigen::Map<rosneuro::DynamicMatrix<float>>(ptr_in, this->nchannels_ - 1, this->chunkSize_);
        Eigen::MatrixXf eog_data = Eigen::Map<Eigen::Matrix<float, 1, -1>>(ptr_eog, 1, this->chunkSize_);
        this->data_in_ = Eigen::MatrixXf(this->nchannels_, this->chunkSize_); 

        this->data_in_.block(0, 0, this->nchannels_-1, this->chunkSize_) = eeg_data;
        this->data_in_.row(this->nchannels_-1) = eog_data;
    }
    this->seq_id_ = msg.header.seq;
}

void Fbcsp::set_message(const Eigen::MatrixXd& data) {
    Eigen::MatrixXf data_float = data.cast<float>(); // [components x bands]
    this->out_.data.resize(data_float.size()); 
    memcpy(this->out_.data.data(),   
           data_float.data(),     
           data_float.size() * sizeof(float));

    this->out_.bands.clear();
    for(const auto& band : this->filters_band_){
        this->out_.bands.insert(this->out_.bands.end(), band.begin(), band.end());
    }

    this->out_.nbands = this->filters_band_.size();
    this->out_.header.stamp = ros::Time::now();
    this->out_.seq = this->seq_id_;
    this->out_.ncomponents = this->ncomponents_;
}

Fbcsp::ApplyResults Fbcsp::apply(void) {
    try {
        Eigen::MatrixXd current_data = this->data_in_.cast<double>(); // [channels x samples]
        
        if (this->do_ica_) {
            current_data = this->ica_matrix_ * current_data;
        }

        Eigen::MatrixXd car_data; // will be [samples x channels]
        if (this->do_car_) {
            car_data = this->car_filter_.apply(current_data.transpose()); 
        } else {
            car_data = current_data.transpose();
        }

        for(int i = 0; i < this->nfilters_; i++) {
            Eigen::MatrixXd data1 = this->filters_low_[i].apply(car_data);
            Eigen::MatrixXd data2 = this->filters_high_[i].apply(data1);
            this->buffers_[i]->add(data2.cast<float>()); // [samples x channels]
        }

        if(!this->buffers_[0]->isfull()) {
            return Fbcsp::ApplyResults::BufferNotFull;
        }

        Eigen::MatrixXd all_processed_signals(this->ncomponents_, this->nfilters_); // [components x bands]

        // FBCSP spatial filter and variance
        for(int i = 0; i < this->nfilters_; i++) {
            Eigen::MatrixXf data_buffer = this->buffers_[i]->get(); // [samples x channels]
            
            // data_buffer is samples x channels
            // csp_matrices_[i] is components x channels
            // result is samples x components
            Eigen::MatrixXf csp_data = data_buffer * this->csp_matrices_[i].transpose().cast<float>();

            // Calculate variance across samples
            Eigen::MatrixXf centered = csp_data.rowwise() - csp_data.colwise().mean();
            Eigen::VectorXf variance = centered.colwise().squaredNorm() / (centered.rows() - 1);

            all_processed_signals.col(i) = variance.cast<double>();
        }

        this->set_message(all_processed_signals);

        return Fbcsp::ApplyResults::Success;

    } catch(std::exception& e) {
        ROS_ERROR("[%s] Error in FBCSP processing: %s", this->name_.c_str(), e.what());
        return Fbcsp::ApplyResults::Error;
    }
}

}
