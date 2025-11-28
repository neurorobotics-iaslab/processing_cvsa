#include "processing_cvsa/CVSA.hpp"

namespace processing{

CVSA::CVSA(void) : nh_("~") { 
    this->pub_ = this->nh_.advertise<processing_cvsa::eeg_power>("/cvsa/eeg_power", 1);
    this->sub_ = this->nh_.subscribe("/neurodata", 1, &processing::CVSA::on_received_data, this);

    this->has_new_data_ = false;
    this->is_configured_ = false;
}

CVSA::CVSA(int nchannels, int frameSize, int bufferSize, int filterOrder, int sampleRate, std::string band_str){
    this->nchannels_ = nchannels;
    this->chunkSize_ = frameSize;
    this->modality_ = "";
    

    if(!str2vecOfvec<float>(band_str, this->filters_band_)){
        throw std::runtime_error("[CVSA Processing] Error in 'filters_band' parameter");
    }
    for(int i = 0; i < this->filters_band_.size(); i++){
        this->filters_low_.push_back(rosneuro::Butterworth<double>(rosneuro::ButterType::LowPass,  filterOrder,  this->filters_band_[i][1], sampleRate));
        this->filters_high_.push_back(rosneuro::Butterworth<double>(rosneuro::ButterType::HighPass,  filterOrder,  this->filters_band_[i][0], sampleRate));
    }
    this->nfilters_ = this->filters_band_.size();
    for (int i = 0; i < this->nfilters_; i++){
        
        this->buffers_.push_back(new rosneuro::RingBuffer<float>());
        if(!this->buffers_.back()->configure("RingBufferCfg")){
            std::ostringstream error_msg;
            error_msg << "[" << this->buffers_.back()->name() << " " 
                  << std::fixed << std::setprecision(2) // Opzionale: per avere 2 decimali
                  << this->filters_band_[i][0] << "-" << this->filters_band_[i][1] << " Hz] "
                  << "Buffer not configured correctly";
            throw std::runtime_error(error_msg.str());
        }
    }

    // fftw configuration for hilbert
    this->fft_buffer_size_ = bufferSize;
    this->fft_in_   = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_buffer_size_);
    this->fft_freq_ = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_buffer_size_);
    this->fft_out_  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_buffer_size_);
    
    this->plan_fwd_ = fftw_plan_dft_1d(fft_buffer_size_, fft_in_, fft_freq_, 
                                 FFTW_FORWARD, FFTW_ESTIMATE);
    this->plan_bwd_ = fftw_plan_dft_1d(fft_buffer_size_, fft_freq_, fft_out_, 
                                 FFTW_BACKWARD, FFTW_ESTIMATE);

    this->is_configured_ = true;
}

CVSA::~CVSA(){
    fftw_destroy_plan(this->plan_fwd_);
    fftw_destroy_plan(this->plan_bwd_);
    fftw_free(this->fft_in_);
    fftw_free(this->fft_freq_);
    fftw_free(this->fft_out_);

    for(auto& buf : this->buffers_){
        delete buf;
    }
}


bool CVSA::configure(void){

    if(ros::param::get("~nchannels", this->nchannels_) == false){
        ROS_ERROR("[CVSA Processing] Missing 'nchannels' parameter, which is a mandatory parameter");
        return false;
    }
    if(ros::param::get("~chunkSize", this->chunkSize_) == false){
        ROS_ERROR("[CVSA Processing] Missing 'chunkSize' parameter, which is a mandatory parameter");
        return false;
    }
    if(ros::param::get("~modality", this->modality_) == false){
        ROS_ERROR("[CVSA Processing] Missing 'modality' parameter, which is a mandatory parameter");
        return false;
    }

    // Filters parameters
    int filterOrder;
    float sampleRate;
    std::string band_str;
    if(ros::param::get("~filter_order", filterOrder) == false){
        ROS_ERROR("[CVSA Processing] Missing 'filter_order' parameter, which is a mandatory parameter");
        return false;
    }
    if(ros::param::get("~samplerate", sampleRate) == false){
        ROS_ERROR("[CVSA Processing] Missing 'sample_rate' parameter, which is a mandatory parameter");
        return false;
    }
    if(ros::param::get("~filters_band", band_str) == false){
        ROS_ERROR("[CVSA Processing] Missing 'filters_band' parameter, which is a mandatory parameter");
        return false;
    }
    if(!str2vecOfvec<float>(band_str, this->filters_band_)){
        ROS_ERROR("[CVSA Processing] Error in 'filters_band' parameter");
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

    this->is_configured_ = true;
    return true;
}

void CVSA::run(){
    ros::Rate r(512);
    if(this->is_configured_ == false){
        ROS_ERROR("[CSVA processing] CVSA not configured correctly");
        return;
    }

    while(ros::ok()){
        if(this->has_new_data_){
            CVSA::ApplyResults res = this->apply();
            this->has_new_data_ = false;
            
            if(res == CVSA::ApplyResults::Error){
                ROS_ERROR("[CSVA processing] Error in CVSA processing");
                break;
            }else if(res == CVSA::ApplyResults::BufferNotFull){
                ROS_WARN("[CSVA processing] Buffer not full");
                this->set_message(Eigen::MatrixXd::Ones(this->nchannels_, this->filters_low_.size())); // no error for the log transform
            }

            this->pub_.publish(this->out_);
        }
        ros::spinOnce();
        r.sleep();
    }
}

void CVSA::on_received_data(const rosneuro_msgs::NeuroFrame &msg){
    this->has_new_data_ = true;

    float* ptr_in;
    float* ptr_eog;
    ptr_in = const_cast<float*>(msg.eeg.data.data());
    ptr_eog = const_cast<float*>(msg.exg.data.data());
    
    if(this->modality_ == "online"){ // reminder: if EOG the last channel is mapped in the exg
        this->data_in_ = Eigen::Map<rosneuro::DynamicMatrix<float>>(ptr_in, this->nchannels_, this->chunkSize_); // channels x sample
    }else if(this->modality_ == "offline"){
        Eigen::MatrixXf eeg_data = Eigen::Map<rosneuro::DynamicMatrix<float>>(ptr_in, this->nchannels_ - 1, this->chunkSize_); // for the eog
        Eigen::MatrixXf eog_data = Eigen::Map<Eigen::Matrix<float, 1, -1>>(ptr_eog, 1, this->chunkSize_);
        this->data_in_ = Eigen::MatrixXf(this->nchannels_, this->chunkSize_); // channels x sample

        // only the last channel is classified as eog (even if it is wrong, since the eog channel is the 18 in py notation)
        this->data_in_.block(0, 0, this->nchannels_-1, this->chunkSize_) = eeg_data;
        this->data_in_.row(this->nchannels_-1) = eog_data;
    }
    this->seq_id_ = msg.header.seq;
}

void CVSA::set_message(Eigen::MatrixXd data){
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

CVSA::ApplyResults CVSA::apply(void){

    Eigen::MatrixXd data1, data2;

    for(int i = 0; i < this->nfilters_; i++){
        data1 = this->filters_low_[i].apply(this->data_in_.transpose().cast<double>());
        data2 = this->filters_high_[i].apply(data1);
        this->buffers_[i]->add(data2.cast<float>()); // [samples x channels]
    }
    if(!this->buffers_[0]->isfull()){
        return CVSA::ApplyResults::BufferNotFull;
    }

    try{
         // [samples x channels]
        Eigen::MatrixXd all_processed_signals(this->nchannels_, this->filters_low_.size()); // [channels x bands]

        // iterate over all filters
        for(int i = 0; i < this->nfilters_; i++){

            Eigen::MatrixXf data_buffer = this->buffers_[i]->get();
            // Bandpass filter
            Eigen::Matrix<double, 1, Eigen::Dynamic> final_data;
            
            // Hibert to compute the power
            Eigen::MatrixXcd analytic_signal = this->compute_analytic_signal(data_buffer.cast<double>());
            data2 = analytic_signal.array().abs2();;

            // Average window 
            final_data = data2.colwise().mean();

            all_processed_signals.col(i) = final_data.transpose();
        }

        // send all the data to the classifier (nbands x nchannels)
        this->set_message(all_processed_signals);

        return CVSA::ApplyResults::Success;

    }catch(std::exception& e){
        ROS_ERROR("[CSVA processing] Error in CVSA processing: %s", e.what());
        return CVSA::ApplyResults::Error;
    }
}

Eigen::MatrixXd CVSA::apply(Eigen::MatrixXf data_in){
    rosneuro::DynamicMatrix<float> tmp_matrix = data_in; 
    Eigen::MatrixXd data1, data2;
    for(int i = 0; i < this->nfilters_; i++){
        data2 = this->filters_low_[i].apply(tmp_matrix.cast<double>());
        data1 = this->filters_high_[i].apply(data2);
        this->buffers_[i]->add(data1.cast<float>()); // [samples x channels]
    }

    if(!this->buffers_[0]->isfull()){
        return Eigen::MatrixXd();
    }

    try{
        Eigen::MatrixXd all_processed_signals(this->nchannels_, this->filters_low_.size()); // [channels x bands]
        std::vector<std::vector<float>> bands;

        // iterate over all filters
        for(int i = 0; i < this->filters_low_.size(); i++){
            // Bandpass filter
            Eigen::Matrix<double, 1, Eigen::Dynamic> final_data;
            Eigen::MatrixXf data_buffer = this->buffers_[i]->get(); // [samples x channels]
            
            // Hibert to compute the power
            Eigen::MatrixXcd analytic_signal = this->compute_analytic_signal(data_buffer.cast<double>());
            data2 = analytic_signal.array().abs2();;

            // Average window 
            final_data = data2.colwise().mean();

            all_processed_signals.col(i) = final_data.transpose();
            bands.push_back({this->filters_band_[i][0], this->filters_band_[i][1]});

        }

        return all_processed_signals;

    }catch(std::exception& e){
        throw std::runtime_error("[CSVA processing] Error in CVSA processing: " + std::string(e.what()));
    }
}

Eigen::MatrixXcd CVSA::compute_analytic_signal(const Eigen::MatrixXd& data){
    int nrows = data.rows();
    int nchannels = data.cols();
    
    // Check if buffer size matches FFT plan size
    if (nrows != this->fft_buffer_size_) {
        ROS_ERROR("[CSVA processing] Data size (%d) does not match FFTW plan size (%d).", nrows, this->fft_buffer_size_);
        throw std::runtime_error("[CSVA processing] Data size does not match FFTW plan size.");
    }
    
    Eigen::MatrixXcd analytic = Eigen::MatrixXcd(nrows, nchannels);

    for (int j = 0; j < nchannels; ++j) {
        // 1. Copy data into FFTW input
        for (int i = 0; i < nrows; ++i) {
            this->fft_in_[i][0] = data(i, j);
            this->fft_in_[i][1] = 0.0;
        }

        // 2. Execute Forward FFT
        fftw_execute_dft(this->plan_fwd_, this->fft_in_, this->fft_freq_);

        // 3. Modify Spectrum (Zero negative, Double positive)
        for (int i = 1; i < nrows / 2; ++i) { // Double positive
            this->fft_freq_[i][0] *= 2.0;
            this->fft_freq_[i][1] *= 2.0;
        }
        for (int i = nrows / 2 + 1; i < nrows; ++i) { // Zero negative
            this->fft_freq_[i][0] = 0.0;
            this->fft_freq_[i][1] = 0.0;
        }
        // Note: DC (0) and Nyquist (nrows/2) components are left unchanged.

        // 4. Execute Inverse FFT
        fftw_execute_dft(this->plan_bwd_, this->fft_freq_, this->fft_out_);

        // 5. Copy to Eigen matrix and normalize by nrows
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