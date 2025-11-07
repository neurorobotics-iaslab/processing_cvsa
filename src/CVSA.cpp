#include "processing_cvsa/CVSA.hpp"

namespace processing{

CVSA::CVSA(void) : nh_("~") { 
    this->pub_ = this->nh_.advertise<processing_cvsa::features>("/cvsa/eeg_power", 1);
    this->sub_ = this->nh_.subscribe("/neurodata", 1, &processing::CVSA::on_received_data, this);

    this->buffer_ = new rosneuro::RingBuffer<float>();
    this->has_new_data_ = false;
}

CVSA::CVSA(int nchannels, int nsamples, int bufferSize, int filterOrder, int sampleRate, std::string band_str){
    this->nchannels_ = nchannels;
    this->nsamples_ = nsamples;
    this->modality_ = "";
    this->buffer_ = new rosneuro::RingBuffer<float>();

    if(!str2vecOfvec<float>(band_str, this->filters_band_)){
        ROS_ERROR("[Processing] Error in 'filters_band' parameter");
    }

    for(int i = 0; i < this->filters_band_.size(); i++){
        this->filters_low_.push_back(rosneuro::Butterworth<double>(rosneuro::ButterType::LowPass,  filterOrder,  this->filters_band_[i][1], sampleRate));
        this->filters_high_.push_back(rosneuro::Butterworth<double>(rosneuro::ButterType::HighPass,  filterOrder,  this->filters_band_[i][0], sampleRate));
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

}

CVSA::~CVSA(){
    fftw_destroy_plan(this->plan_fwd_);
    fftw_destroy_plan(this->plan_bwd_);
    fftw_free(this->fft_in_);
    fftw_free(this->fft_freq_);
    fftw_free(this->fft_out_);

    delete this->buffer_;
}


bool CVSA::configure(void){

    if(ros::param::get("~nchannels", this->nchannels_) == false){
        ROS_ERROR("[Processing] Missing 'nchannels' parameter, which is a mandatory parameter");
        return false;
    }
    if(ros::param::get("~nsamples", this->nsamples_) == false){
        ROS_ERROR("[Processing] Missing 'nsamples' parameter, which is a mandatory parameter");
        return false;
    }
    if(ros::param::get("~modality", this->modality_) == false){
        ROS_ERROR("[Processing] Missing 'modality' parameter, which is a mandatory parameter");
        return false;
    }

    // Buffer configuration -> TODO: create the yaml to load with buffer parameters
    if(!this->buffer_->configure("RingBufferCfg")){
        ROS_ERROR("[%s] Buffer not configured correctly", this->buffer_->name().c_str());
        return false;
    }

    // Filters parameters
    int filterOrder;
    float sampleRate;
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
    if(!str2vecOfvec<float>(band_str, this->filters_band_)){
        ROS_ERROR("[Processing] Error in 'filters_band' parameter");
        return false;
    }

    // Filter configuration
    for(int i = 0; i < this->filters_band_.size(); i++){
        this->filters_low_.push_back(rosneuro::Butterworth<double>(rosneuro::ButterType::LowPass,  filterOrder,  this->filters_band_[i][1], sampleRate));
        this->filters_high_.push_back(rosneuro::Butterworth<double>(rosneuro::ButterType::HighPass,  filterOrder,  this->filters_band_[i][0], sampleRate));
    }

    // fftw configuration for hilbert
    this->buffer_->getParam(std::string("size"), this->fft_buffer_size_);
    this->fft_in_   = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_buffer_size_);
    this->fft_freq_ = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_buffer_size_);
    this->fft_out_  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_buffer_size_);
    
    this->plan_fwd_ = fftw_plan_dft_1d(fft_buffer_size_, fft_in_, fft_freq_, 
                                 FFTW_FORWARD, FFTW_ESTIMATE);
    this->plan_bwd_ = fftw_plan_dft_1d(fft_buffer_size_, fft_freq_, fft_out_, 
                                 FFTW_BACKWARD, FFTW_ESTIMATE);

    return true;
}

void CVSA::run(){
    ros::Rate r(512);

    while(ros::ok()){
        if(this->has_new_data_){
            CVSA::ClassifyResults res = this->apply();
            this->has_new_data_ = false;
            
            if(res == CVSA::ClassifyResults::Error){
                ROS_ERROR("[CSVA processing] Error in CVSA processing");
                break;
            }else if(res == CVSA::ClassifyResults::BufferNotFull){
                ROS_WARN("[CSVA processing] Buffer not full");
                continue;
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
        this->data_in_ = Eigen::Map<rosneuro::DynamicMatrix<float>>(ptr_in, this->nchannels_, this->nsamples_); // channels x sample
    }else if(this->modality_ == "offline"){
        Eigen::MatrixXf eeg_data = Eigen::Map<rosneuro::DynamicMatrix<float>>(ptr_in, this->nchannels_ - 1, this->nsamples_); // for the eog
        Eigen::MatrixXf eog_data = Eigen::Map<Eigen::Matrix<float, 1, -1>>(ptr_eog, 1, this->nsamples_);
        this->data_in_ = Eigen::MatrixXf(this->nchannels_, this->nsamples_); // channels x sample

        // only the last channel is classified as eog (even if it is wrong, since the eog channel is the 18 in py notation)
        this->data_in_.block(0, 0, this->nchannels_-1, this->nsamples_) = eeg_data;
        this->data_in_.row(this->nchannels_-1) = eog_data;
    }
}

void CVSA::set_message(std::vector<Eigen::Matrix<double, 1, Eigen::Dynamic>> data, std::vector<std::vector<float>> filters_band){
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

CVSA::ClassifyResults CVSA::apply(void){

    this->buffer_->add(this->data_in_.transpose().cast<float>()); // [samples x channels]
    if(!this->buffer_->isfull()){
        return CVSA::ClassifyResults::BufferNotFull;
    }

    try{
        Eigen::MatrixXf data_buffer = this->buffer_->get(); // [samples x channels]
        std::vector<Eigen::Matrix<double, 1, Eigen::Dynamic>> all_processed_signals;
        std::vector<std::vector<float>> bands;

        // iterate over all filters
        for(int i = 0; i < this->filters_low_.size(); i++){
            // Bandpass filter
            Eigen::MatrixXd data1, data2;
            Eigen::Matrix<double, 1, Eigen::Dynamic> final_data;
            data2 = this->filters_low_[i].apply(data_buffer.cast<double>());
            data1 = this->filters_high_[i].apply(data2);
            
            // Hibert to compute the power
            Eigen::MatrixXcd analytic_signal = this->compute_analytic_signal(data1);
            data2 = analytic_signal.array().abs2();;

            // Average window 
            final_data = data2.colwise().mean();

            all_processed_signals.push_back(final_data);
            bands.push_back({this->filters_band_[i][0], this->filters_band_[i][1]});

        }

        // send all the data to the classifier (nbands x nchannels)
        this->set_message(all_processed_signals, bands);

        return CVSA::ClassifyResults::Success;

    }catch(std::exception& e){
        ROS_ERROR("[CSVA processing] Error in CVSA processing: %s", e.what());
        return CVSA::ClassifyResults::Error;
    }
}

std::vector<Eigen::Matrix<double, 1, Eigen::Dynamic>> CVSA::apply(Eigen::MatrixXf data_in){

    try{
        Eigen::MatrixXf data_buffer = data_in; // [samples x channels]
        std::vector<Eigen::Matrix<double, 1, Eigen::Dynamic>> all_processed_signals;
        std::vector<std::vector<float>> bands;

        // iterate over all filters
        for(int i = 0; i < this->filters_low_.size(); i++){
            // Bandpass filter
            Eigen::MatrixXd data1, data2;
            Eigen::Matrix<double, 1, Eigen::Dynamic> final_data;
            data2 = this->filters_low_[i].apply(data_buffer.cast<double>());
            data1 = this->filters_high_[i].apply(data2);
            
            // Hibert to compute the power
            Eigen::MatrixXcd analytic_signal = this->compute_analytic_signal(data1);
            data2 = analytic_signal.array().abs2();;

            // Average window 
            final_data = data2.colwise().mean();

            all_processed_signals.push_back(final_data);
            bands.push_back({this->filters_band_[i][0], this->filters_band_[i][1]});

        }

        return all_processed_signals;

    }catch(std::exception& e){
        throw std::runtime_error("Error in CVSA processing: " + std::string(e.what()));
    }
}

Eigen::MatrixXcd CVSA::compute_analytic_signal(const Eigen::MatrixXd& data){
    int nrows = data.rows();
    int nchannels = data.cols();
    
    // Check if buffer size matches FFT plan size
    if (nrows != this->fft_buffer_size_) {
        throw std::runtime_error("Data size does not match FFTW plan size.");
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