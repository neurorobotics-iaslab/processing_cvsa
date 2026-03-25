#include "processing_bci/Power.hpp"

namespace processing{

Power::Power(void) : nh_("~") { 
    this->name_ = "Power Processing";
    this->sub_ = this->nh_.subscribe("/neurodata", 1, &processing::Power::on_received_data, this);

    this->has_new_data_ = false;
    this->is_configured_ = false;
}

Power::~Power(){
    fftw_destroy_plan(this->plan_fwd_);
    fftw_destroy_plan(this->plan_bwd_);
    fftw_free(this->fft_in_);
    fftw_free(this->fft_freq_);
    fftw_free(this->fft_out_);

    for(auto& buf : this->buffers_){
        delete buf;
    }
}


bool Power::configure(void){

    if(ros::param::get("~nchannels", this->nchannels_) == false){
        ROS_ERROR("[%s] Missing 'nchannels' parameter, which is a mandatory parameter", this->name_.c_str());
        return false;
    }
    if(ros::param::get("~signal_type", this->signal_type_) == false){
        ROS_ERROR("[%s] Missing 'signal_type_' parameter, which is a mandatory parameter", this->name_.c_str());
        return false;
    }
    if(ros::param::get("~chunkSize", this->chunkSize_) == false){
        ROS_ERROR("[%s] Missing 'chunkSize' parameter, which is a mandatory parameter", this->name_.c_str());
        return false;
    }
    if(ros::param::get("~run_mode", this->run_mode_) == false){
        ROS_ERROR("[%s] Missing 'run_mode' parameter, which is a mandatory parameter", this->name_.c_str());
        return false;
    }
    std::string topic_to_pub;
    this->nh_.param<std::string>("topic_to_pub", topic_to_pub, "/eeg_power");
    ROS_INFO("[%s] topic_to_pub set to: %s", this->name_.c_str(), topic_to_pub.c_str());
    this->pub_ = this->nh_.advertise<processing_bci::eeg_power>(topic_to_pub, 1);

    // Filters parameters
    int filterOrder;
    float sampleRate;
    std::string band_str;
    if(ros::param::get("~filter_order", filterOrder) == false){
        ROS_ERROR("[%s] Missing 'filter_order' parameter, which is a mandatory parameter", this->name_.c_str());
        return false;
    }
    if(ros::param::get("~samplerate", sampleRate) == false){
        ROS_ERROR("[%s] Missing 'sample_rate' parameter, which is a mandatory parameter", this->name_.c_str());
        return false;
    }
    if(ros::param::get("~filters_band", band_str) == false){
        ROS_ERROR("[%s] Missing 'filters_band' parameter, which is a mandatory parameter", this->name_.c_str());
        return false;
    }
    if(!str2vecOfvec<float>(band_str, this->filters_band_)){
        ROS_ERROR("[%s] Error in 'filters_band' parameter", this->name_.c_str());
        return false;
    }
    this->nfilters_ = this->filters_band_.size();

    // Filter configuration
    for(int i = 0; i < this->filters_band_.size(); i++){
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
        }
    }

    // fftw configuration for hilbert
    this->buffers_[0]->getParam(std::string("size"), this->fft_buffer_size_);
    this->fft_in_   = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_buffer_size_);
    this->fft_freq_ = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_buffer_size_);
    this->fft_out_  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_buffer_size_);
    
    this->plan_fwd_ = fftw_plan_dft_1d(fft_buffer_size_, fft_in_, fft_freq_, 
                                 FFTW_FORWARD, FFTW_ESTIMATE);
    this->plan_bwd_ = fftw_plan_dft_1d(fft_buffer_size_, fft_freq_, fft_out_, 
                                 FFTW_BACKWARD, FFTW_ESTIMATE);

    this->car_filter_ = rosneuro::Car<double>();
    this->car_filter_.configure("CarCfg");

    this->is_configured_ = true;
    return true;
}

void Power::run(){
    ros::Rate r(512);
    if(this->is_configured_ == false){
        ROS_ERROR("[%s] Power not configured correctly", this->name_.c_str());
        return;
    }

    while(ros::ok()){
        if(this->has_new_data_){
            Power::ApplyResults res = this->apply();
            this->has_new_data_ = false;
            
            if(res == Power::ApplyResults::Error){
                ROS_ERROR("[%s] Error in Power processing", this->name_.c_str());
                break;
            }else if(res == Power::ApplyResults::BufferNotFull){
                ROS_WARN("[%s] Buffer not full", this->name_.c_str());
                this->set_message(Eigen::MatrixXd::Ones(this->nchannels_, this->filters_low_.size())); // no error for the log transform
            }

            this->pub_.publish(this->out_);
        }
        ros::spinOnce();
        r.sleep();
    }
}

void Power::on_received_data(const rosneuro_msgs::NeuroFrame &msg){
    this->has_new_data_ = true;

    float* ptr_in;
    float* ptr_eog;
    ptr_in = const_cast<float*>(msg.eeg.data.data());
    ptr_eog = const_cast<float*>(msg.exg.data.data());
    
    if(this->run_mode_ == "online"  || (this->run_mode_ == "offline" && this->signal_type_ == "eeg")){ // reminder: if EOG the last channel is mapped in the exg
        this->data_in_ = Eigen::Map<rosneuro::DynamicMatrix<float>>(ptr_in, this->nchannels_, this->chunkSize_); // channels x sample
    }else if(this->run_mode_ == "offline" && this->signal_type_ == "eeg_eog"){
        Eigen::MatrixXf eeg_data = Eigen::Map<rosneuro::DynamicMatrix<float>>(ptr_in, this->nchannels_ - 1, this->chunkSize_); // for the eog
        Eigen::MatrixXf eog_data = Eigen::Map<Eigen::Matrix<float, 1, -1>>(ptr_eog, 1, this->chunkSize_);
        this->data_in_ = Eigen::MatrixXf(this->nchannels_, this->chunkSize_); // channels x sample

        // only the last channel is classified as eog (even if it is wrong, since the eog channel is the 18 in py notation)
        this->data_in_.block(0, 0, this->nchannels_-1, this->chunkSize_) = eeg_data;
        this->data_in_.row(this->nchannels_-1) = eog_data;
    }
    this->seq_id_ = msg.header.seq;
}

void Power::set_message(Eigen::MatrixXd data){
    // flattering data in column major order.
    Eigen::MatrixXf data_float = data.cast<float>(); // [channels x bands]
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
    this->out_.nchannels = this->nchannels_;
}

Power::ApplyResults Power::apply(void){

    Eigen::MatrixXd data1, data2;
    Eigen::MatrixXd car_data;

    car_data = this->car_filter_.apply(this->data_in_.transpose().cast<double>()); // [samples x channels]

    for(int i = 0; i < this->nfilters_; i++){
        data1 = this->filters_low_[i].apply(car_data);
        data2 = this->filters_high_[i].apply(data1);
        this->buffers_[i]->add(data2.cast<float>()); // [samples x channels]
    }
    if(!this->buffers_[0]->isfull()){
        return Power::ApplyResults::BufferNotFull;
    }

    try{
         // [samples x channels]
        Eigen::MatrixXd all_processed_signals(this->nchannels_, this->filters_low_.size()); // [channels x bands]

        // iterate over all filters
        for(int i = 0; i < this->nfilters_; i++){

            Eigen::MatrixXf data_buffer = this->buffers_[i]->get();
            // Bandpass filter
            Eigen::Matrix<double, 1, Eigen::Dynamic> final_data;
            
            // Hibert to compute the Power
            Eigen::MatrixXcd analytic_signal = this->compute_analytic_signal(data_buffer.cast<double>());
            data2 = analytic_signal.array().abs2();;

            // Average window 
            final_data = data2.colwise().mean();

            all_processed_signals.col(i) = final_data.transpose();
        }

        // send all the data to the classifier (nbands x nchannels)
        this->set_message(all_processed_signals);

        return Power::ApplyResults::Success;

    }catch(std::exception& e){
        ROS_ERROR("[%s] Error in Power processing: %s", this->name_.c_str(), e.what());
        return Power::ApplyResults::Error;
    }
}

Eigen::MatrixXcd Power::compute_analytic_signal(const Eigen::MatrixXd& data){
    int nrows = data.rows();
    int nchannels = data.cols();
    
    // Check if buffer size matches FFT plan size
    if (nrows != this->fft_buffer_size_) {
        ROS_ERROR("[%s] Data size (%d) does not match FFTW plan size (%d).", this->name_.c_str(), nrows, this->fft_buffer_size_);
        throw std::runtime_error("[Power Processing] Data size does not match FFTW plan size.");
    }
    
    Eigen::MatrixXcd analytic = Eigen::MatrixXcd(nrows, nchannels);

    for (int j = 0; j < nchannels; ++j) {
        for (int i = 0; i < nrows; ++i) {
            this->fft_in_[i][0] = data(i, j);
            this->fft_in_[i][1] = 0.0;
        }

        // Execute Forward FFT
        fftw_execute_dft(this->plan_fwd_, this->fft_in_, this->fft_freq_);

        // Modify Spectrum (Zero negative, Double positive)
        for (int i = 1; i < nrows / 2; ++i) { // Double positive
            this->fft_freq_[i][0] *= 2.0;
            this->fft_freq_[i][1] *= 2.0;
        }
        for (int i = nrows / 2 + 1; i < nrows; ++i) { // Zero negative
            this->fft_freq_[i][0] = 0.0;
            this->fft_freq_[i][1] = 0.0;
        }

        // Execute Inverse FFT
        fftw_execute_dft(this->plan_bwd_, this->fft_freq_, this->fft_out_);

        for (int i = 0; i < nrows; ++i) {
            analytic(i, j) = std::complex<double>(
                this->fft_out_[i][0] / nrows, // Real part
                this->fft_out_[i][1] / nrows  // Imaginary (Hilbert Tx)
            );
        }
    }
    return analytic;
}
}