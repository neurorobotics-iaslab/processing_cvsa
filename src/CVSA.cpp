#include "processing_cvsa/CVSA.hpp"

namespace processing{

CVSA::CVSA(void) : nh_("~") { 
    this->pub_ = this->nh_.advertise<rosneuro_msgs::NeuroOutput>("/cvsa/neuroprediction", 1);
    this->sub_ = this->nh_.subscribe("/neurodata", 1, &processing::CVSA::on_received_data, this);

    this->buffer_ = new rosneuro::RingBuffer<float>();
    this->decoder_ = new rosneuro::decoder::Decoder();
    this->has_new_data_ = false;
}

CVSA::~CVSA(){
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

    // Buffer configuration -> TODO: create the yaml to load with buffer parameters
    if(!this->buffer_->configure("RingBufferCfg")){
        ROS_ERROR("[%s] Buffer not configured correctly", this->buffer_->name().c_str());
        return false;
    }

    // Decoder configuration
    if(!this->decoder_->configure()){
        ROS_ERROR("[%s] decoder not confgured correctly", this->decoder_->name().c_str());
        return false;
    }
    this->idchans_features_ = this->decoder_->get_idchans();
    this->features_band_ = this->decoder_->get_bands();

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
    if(!str2vecOfvec<float>(band_str, this->filters_band_)){
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
        this->filters_low_.push_back(rosneuro::Butterworth<float>(rosneuro::ButterType::LowPass,  filterOrder,  this->filters_band_[i][1], sampleRate));
        this->filters_high_.push_back(rosneuro::Butterworth<float>(rosneuro::ButterType::HighPass,  filterOrder,  this->filters_band_[i][0], sampleRate));
    }

    /* if you want to use the yaml files
    if(this->filter_low_.configure()){
        ROS_INFO("Filter low configured");
    }else{
        ROS_ERROR("Filter low not configured");
        return false;
    }
    */
    return true;
}

void CVSA::run(){
    ros::Rate r(512);

    while(ros::ok()){
        if(this->has_new_data_){
            if(this->classify() == CVSA::ClassifyResults::Error){
                ROS_ERROR("Error in CVSA processing");
                break;
            }else if(this->classify() == CVSA::ClassifyResults::BufferNotFull){
                ROS_WARN("Buffer not full");
                continue;
            }

            this->set_message();
            this->pub_.publish(this->out_);
            this->has_new_data_ = false;
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
    this->data_in_ = Eigen::Map<rosneuro::DynamicMatrix<float>>(ptr_in, this->nchannels_, this->nsamples_);
    this->out_.neuroheader = msg.neuroheader;
}

void CVSA::set_message(void){
    this->out_.header.stamp = ros::Time::now();
	this->out_.softpredict.data = std::vector<float>(this->rawProb_.data(), this->rawProb_.data() + this->rawProb_.rows() * this->rawProb_.cols());
    
}

CVSA::ClassifyResults CVSA::classify(void){
    this->buffer_->add(this->data_in_.transpose().cast<float>()); // [samples x channels]
    if(!this->buffer_->isfull()){
        return CVSA::ClassifyResults::BufferNotFull;
    }

    try{
        Eigen::MatrixXf data_buffer = this->buffer_->get();
        std::vector<Eigen::Matrix<float, 1, Eigen::Dynamic>> all_processed_signals;
        // iterate over all filters
        for(int i = 0; i < this->filters_low_.size(); i++){
            // Bandpass filter
            Eigen::MatrixXf data1, data2;
            Eigen::Matrix<float, 1, Eigen::Dynamic> final_data;
            data1 = this->filters_low_[i].apply(data_buffer);
            data2 = this->filters_high_[i].apply(data1);
            
            // Rectifing signal
            data2 = data1.array().pow(2);

            // Average window 
            data1 = data2.colwise().mean();

            // Logarithm
            final_data = data1.array().log();

            all_processed_signals.push_back(final_data);

        }

        // Extract features
        Eigen::VectorXf features = get_features<float>(all_processed_signals, this->idchans_features_, this->features_band_, this->filters_band_);

        // classify (use the decoder function apply) and save in rawProb_
        //this->rawProb_ = this->decoder_->apply(Eigen::Map<Eigen::VectorXf>(features.data(), features.size()));
        this->rawProb_ = Eigen::VectorXf::Ones(2);

        return CVSA::ClassifyResults::Success;

    }catch(std::exception& e){
        ROS_ERROR("Error in CVSA processing: %s", e.what());
        return CVSA::ClassifyResults::Error;
    }
}
}