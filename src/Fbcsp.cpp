#include "processing_bci/Fbcsp.hpp"
#include <XmlRpcException.h>

namespace processing {

Fbcsp::Fbcsp(void) : nh_("~") { 
    this->name_ = "FBCSP Processing";
    this->sub_ = this->nh_.subscribe("/neurodata", 1, &processing::Fbcsp::on_received_data, this);

    this->has_new_data_ = false;
    this->is_configured_ = false;
    this->is_car_configured_ = false;
    this->csp_ch_idx_resolved_ = false;
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

    std::string paradigm, paradigm_upper;
    if(ros::param::get("~paradigm", paradigm) == false){
        ROS_ERROR("[%s] Missing 'paradigm' parameter", this->name_.c_str());
        return false;
    }
    paradigm_upper = paradigm;
    std::transform(paradigm_upper.begin(), paradigm_upper.end(), paradigm_upper.begin(), [](unsigned char c) {
        return std::toupper(c);
    });
    this->name_ = this->name_  + " " + paradigm_upper;
    std::string topic_to_pub = "/" + paradigm + "/eeg_fbcsp";
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
        ROS_INFO("[%s] CAR filter enabled – channel names resolved on first NeuroFrame", this->name_.c_str());
        XmlRpc::XmlRpcValue eog_names_xml;
        if(!ros::param::get("/CarCfg/params/EOG_ch_names", eog_names_xml) ||
           eog_names_xml.getType() != XmlRpc::XmlRpcValue::TypeArray){
            ROS_ERROR("[%s] Missing or invalid '/CarCfg/params/EOG_ch_names'", this->name_.c_str());
            return false;
        }
        for(int i = 0; i < eog_names_xml.size(); i++){
            this->EOG_ch_names_.push_back(static_cast<std::string>(eog_names_xml[i]));
        }
        ROS_INFO("[%s] EOG channel names to exclude from CAR: %zu channel(s)", this->name_.c_str(), this->EOG_ch_names_.size());
        // actual Car configuration deferred to first NeuroFrame (configure_car())
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

    // CSP Matrices configuration (private namespace: loaded inside <node> in the launch file)
    if (!this->load_matrices("CspCfg/params/csp_matrices", this->csp_matrices_)) {
        ROS_ERROR("[%s] Failed to load 'CspCfg/params/csp_matrices' from parameter server", this->name_.c_str());
        return false;
    }
    if (this->csp_matrices_.size() != this->nfilters_) {
        ROS_ERROR("[%s] Number of CSP matrices (%ld) does not match number of frequency bands (%d)",
                  this->name_.c_str(), this->csp_matrices_.size(), this->nfilters_);
        return false;
    }
    this->ncomponents_ = this->csp_matrices_[0].rows();
    ROS_INFO("[%s] Loaded %ld CSP matrices, %d components per band", this->name_.c_str(), this->csp_matrices_.size(), this->ncomponents_);

    // CSP selected channels (names resolved from NeuroFrame labels on first message)
    XmlRpc::XmlRpcValue csp_ch_xml;
    if(this->nh_.getParam("CspCfg/params/selected_channels", csp_ch_xml) &&
       csp_ch_xml.getType() == XmlRpc::XmlRpcValue::TypeArray){
        for(int i = 0; i < csp_ch_xml.size(); i++)
            this->csp_ch_names_.push_back(static_cast<std::string>(csp_ch_xml[i]));
        ROS_INFO("[%s] CSP channel selection: %zu channel(s) – resolved on first NeuroFrame",
                 this->name_.c_str(), this->csp_ch_names_.size());
    } else {
        ROS_INFO("[%s] No CSP channel selection found – all NeuroFrame channels will be used", this->name_.c_str());
    }

    // csp and filter bands check
    Eigen::MatrixXd yaml_bands_mat;
    if (this->load_matrix("CspCfg/params/bands", yaml_bands_mat)) {
        if (yaml_bands_mat.rows() != this->nfilters_ || yaml_bands_mat.cols() < 2) {
            ROS_ERROR("[%s] Mismatch! Launch file has %d bands, but YAML model has %ld bands.", 
                      this->name_.c_str(), this->nfilters_, yaml_bands_mat.rows());
            return false;
        }
        for (int i = 0; i < this->nfilters_; ++i) {
            if (std::abs(this->filters_band_[i][0] - yaml_bands_mat(i, 0)) > 1e-4 || 
                std::abs(this->filters_band_[i][1] - yaml_bands_mat(i, 1)) > 1e-4) {
                
                ROS_ERROR("[%s] Band mismatch at index %d! Launch file: [%.1f, %.1f] Hz, YAML: [%.1f, %.1f] Hz",
                          this->name_.c_str(), i, this->filters_band_[i][0], this->filters_band_[i][1], 
                          yaml_bands_mat(i, 0), yaml_bands_mat(i, 1));
                return false;
            }
        }
        ROS_INFO("[%s] Success: Frequency bands in Launch file perfectly match the CSP YAML configuration.", this->name_.c_str());
    } else {
        ROS_WARN("[%s] 'cspCfg/params/bands' parameter not found in YAML. Skipping safety check.", this->name_.c_str());
        return false;
    }

    // Filter configuration
    for(int i = 0; i < this->nfilters_; i++){
        this->filters_low_.push_back(rosneuro::Butterworth<double>(rosneuro::ButterType::LowPass,  filterOrder,  this->filters_band_[i][1], sampleRate));
        this->filters_high_.push_back(rosneuro::Butterworth<double>(rosneuro::ButterType::HighPass,  filterOrder,  this->filters_band_[i][0], sampleRate));
    }

    // Buffer configuration 
    for(int i = 0; i < this->nfilters_; i++){
        this->buffers_.push_back(new rosneuro::RingBuffer<double>());
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

    if(this->do_car_ && !this->is_car_configured_){
        const auto& labels = msg.eeg.info.labels;
        if(labels.empty()){
            ROS_ERROR_THROTTLE(5.0, "[%s] NeuroFrame labels empty – cannot configure CAR", this->name_.c_str());
            return;
        }
        this->EOG_ch_.clear();
        for(const auto& name : this->EOG_ch_names_){
            bool found = false;
            for(int i = 0; i < static_cast<int>(labels.size()); i++){
                std::string a = name, b = labels[i];
                std::transform(a.begin(), a.end(), a.begin(), ::tolower);
                std::transform(b.begin(), b.end(), b.begin(), ::tolower);
                if(a == b){ this->EOG_ch_.push_back(i); found = true; break; }
            }
            if(!found){
                ROS_ERROR("[%s] CAR: channel '%s' not found in NeuroFrame labels", this->name_.c_str(), name.c_str());
                return;
            }
        }
        this->car_filter_ = rosneuro::Car<double>();
        this->car_filter_.configure(this->EOG_ch_);
        this->is_car_configured_ = true;
        ROS_INFO("[%s] CAR configured with %zu excluded channel(s)", this->name_.c_str(), this->EOG_ch_.size());
    }

    // Resolve CSP channel indices on first message
    if(!this->csp_ch_idx_resolved_){
        const auto& labels = msg.eeg.info.labels;
        if(this->csp_ch_names_.empty()){
            // no selection: use all channels
            for(int i = 0; i < static_cast<int>(msg.eeg.info.nchannels); i++)
                this->csp_ch_idx_.push_back(i);
            this->csp_ch_idx_resolved_ = true;
        } else {
            if(labels.empty()){
                ROS_ERROR_THROTTLE(5.0, "[%s] NeuroFrame labels empty – cannot resolve CSP channels", this->name_.c_str());
                return;
            }
            for(const auto& ch_name : this->csp_ch_names_){
                bool found = false;
                for(int i = 0; i < static_cast<int>(labels.size()); i++){
                    std::string a = ch_name, b = labels[i];
                    std::transform(a.begin(), a.end(), a.begin(), ::tolower);
                    std::transform(b.begin(), b.end(), b.begin(), ::tolower);
                    if(a == b){ this->csp_ch_idx_.push_back(i); found = true; break; }
                }
                if(!found){
                    ROS_ERROR("[%s] CSP channel '%s' not found in NeuroFrame labels", this->name_.c_str(), ch_name.c_str());
                    return;
                }
            }
            // Verify CSP matrix column count matches selected channels
            if(this->csp_matrices_[0].cols() != static_cast<int>(this->csp_ch_idx_.size())){
                ROS_ERROR("[%s] CSP matrix has %ld columns but %zu channels were selected – mismatch",
                          this->name_.c_str(), this->csp_matrices_[0].cols(), this->csp_ch_idx_.size());
                return;
            }
            this->csp_ch_idx_resolved_ = true;
            ROS_INFO("[%s] CSP channels resolved: %zu channel(s)", this->name_.c_str(), this->csp_ch_idx_.size());
        }
    }

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
    this->out_.ncomponents_for_band = this->ncomponents_;
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
            this->buffers_[i]->add(data2); // [samples x channels], double
        }

        if(!this->buffers_[0]->isfull()) {
            return Fbcsp::ApplyResults::BufferNotFull;
        }

        Eigen::MatrixXd all_processed_signals(this->ncomponents_, this->nfilters_); // [components x bands]

        // FBCSP spatial filter and variance (all in double — matches MATLAB exactly)
        for(int i = 0; i < this->nfilters_; i++) {
            Eigen::MatrixXd data_buffer = this->buffers_[i]->get(); // [samples x channels]

            // Extract selected channels: [bufferSize x n_selected]
            int n_sel = this->csp_ch_idx_.size();
            Eigen::MatrixXd buf_selected(data_buffer.rows(), n_sel);
            for(int j = 0; j < n_sel; j++)
                buf_selected.col(j) = data_buffer.col(this->csp_ch_idx_[j]);

            // CSP spatial filter: [bufferSize x n_components]
            // csp_matrices_[i] is [n_components x n_selected]
            Eigen::MatrixXd csp_data = buf_selected * this->csp_matrices_[i].transpose();

            // Mean power: mean(x^2) — matches MNE CSP with log=True
            Eigen::VectorXd variance = csp_data.colwise().squaredNorm() / csp_data.rows();

            all_processed_signals.col(i) = variance;
        }

        this->set_message(all_processed_signals);

        return Fbcsp::ApplyResults::Success;

    } catch(std::exception& e) {
        ROS_ERROR("[%s] Error in FBCSP processing: %s", this->name_.c_str(), e.what());
        return Fbcsp::ApplyResults::Error;
    }
}

}
