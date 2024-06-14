#include "processing_cvsa/CVSA.hpp"

processing::CVSA::CVSA(void) : nh_("~") { 
    this->pub_ = this->nh_.advertise<rosneuro_msgs::NeuroOutput>("/cvsa/neuroprediction", 1);
    this->sub_ = this->nh_.subscribe("/neurodata", 1, &processing::CVSA::on_received_data, this);

    this->buffer_ = new rosneuro::RingBuffer<float>();
    this->decoder_ = new rosneuro::decoder::Decoder();
    this->has_new_data_ = false;
    
}

processing::CVSA::~CVSA(){
    this->outputFile_.close();
}


bool processing::CVSA::configure(void){

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


    const std::string fileoutput = "/home/paolo/cvsa_ws/src/processing_cvsa/src/features_rosneuro.csv";
    this->outputFile_.open(fileoutput);
    if(this->outputFile_.is_open()){
        std::cout << "file opened" << std::endl;
    }
    return true;
}

void processing::CVSA::run(){
    ros::Rate r(512);

    while(ros::ok()){
        if(this->has_new_data_){
            if(!this->classify()){
                this->has_new_data_ = false;
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

void processing::CVSA::on_received_data(const rosneuro_msgs::NeuroFrame &msg){
    this->has_new_data_ = true;

    float* ptr_in;
    float* ptr_eog;
    ptr_in = const_cast<float*>(msg.eeg.data.data());
    ptr_eog = const_cast<float*>(msg.exg.data.data());
    this->data_in_ = Eigen::Map<rosneuro::DynamicMatrix<float>>(ptr_in, this->nchannels_, this->nsamples_);
    this->out_.neuroheader = msg.neuroheader;
}

void processing::CVSA::set_message(void){
    this->out_.header.stamp = ros::Time::now();
	this->out_.softpredict.data = std::vector<float>(this->rawProb_.data(), this->rawProb_.data() + this->rawProb_.rows() * this->rawProb_.cols());
    
}

bool processing::CVSA::classify(void){
    this->buffer_->add(this->data_in_.transpose().cast<float>()); // [samples x channels]
    if(!this->buffer_->isfull()){
        return false;
    }

    // Bandpass -> similar results of filter in matlab
    Eigen::MatrixXf data; // [samples x channels]
    std::vector<double> a, b; 
    a = ComputeDenCoeffs(4, 8.0*2.0/512.0, 14.0*2.0/512.0);
    b = ComputeNumCoeffs(4, 8.0*2.0/512.0, 14.0*2.0/512.0, a);
    data = filter(this->buffer_->get(), b, a);
    
    // Rectifing signal
    data = data.array().pow(2);
    

    // Average window ---------> TODO: correct here
    float avg = 1;
    float windowSize = avg * 512;
    Eigen::VectorXd tmp_b = Eigen::VectorXd::Ones(windowSize) / windowSize;
    std::vector<double> b2(tmp_b.data(), tmp_b.data() + tmp_b.size());
    a.erase(a.begin(), a.end());
    a.push_back(1.0);
    data = filter(data, b2, a);

    // Logarithm
    data = data.array().log();

    // Extract features
    ROS_INFO("Data received: %d x %d", data.rows(), data.cols());
    Eigen::MatrixXf features = this->decoder_->getFeatures(data);
    features.transposeInPlace();

    ROS_INFO("Data received: %d x %d", features.rows(), features.cols());
    if(this->outputFile_.is_open()){
        for(int i = 0; i < features.rows(); i++){
            for(int j = 0; j < features.cols(); j++){
                this->outputFile_ << features(i,j) << " ";
            }
            this->outputFile_ << std::endl;
        }
    }

    // classify (use the decoder function apply) and save in rawProb_
    //this->rawProb_ = this->decoder_->apply(features);

    return true;
}

Eigen::MatrixXf processing::CVSA::filter(const Eigen::MatrixXf& in, const std::vector<double>& coeff_b, const std::vector<double>& coeff_a) {
    Eigen::MatrixXf out(in.rows(), in.cols());
    for(int i = 0; i < in.cols(); i++){
        Eigen::VectorXf x = in.col(i);
    
        int len_x = x.size();
        int len_b = coeff_b.size();
        int len_a = coeff_a.size();

        Eigen::VectorXf zi = Eigen::VectorXf::Zero(len_b);  // Initialize zi with zeros
        Eigen::VectorXf filter_x = Eigen::VectorXf::Zero(len_x);  // Initialize the output with zeros

        if (len_a == 1) {
            for (int m = 0; m < len_x; m++) {
                filter_x[m] = coeff_b[0] * x[m] + zi[0];
                for (int i = 1; i < len_b; i++) {
                    zi[i - 1] = coeff_b[i] * x[m] + zi[i];
                }
            }
        } else {
            for (int m = 0; m < len_x; m++) {
                filter_x[m] = coeff_b[0] * x[m] + zi[0];
                for (int i = 1; i < len_b; i++) {
                    zi[i - 1] = coeff_b[i] * x[m] + zi[i] - coeff_a[i] * filter_x[m];
                }
            }
        }
        out.col(i) = filter_x;
    }

    return out;
}

// b
std::vector<double> processing::CVSA::ComputeNumCoeffs(int FilterOrder, double Lcutoff, double Ucutoff, std::vector<double> DenC){
	std::vector<double> TCoeffs;
	std::vector<double> NumCoeffs(2 * FilterOrder + 1);
	std::vector<std::complex<double>> NormalizedKernel(2 * FilterOrder + 1);

	std::vector<double> Numbers;
	for (double n = 0; n < FilterOrder * 2 + 1; n++)
		Numbers.push_back(n);
	int i;

	TCoeffs = ComputeHP(FilterOrder);

	for (i = 0; i < FilterOrder; ++i)
	{
		NumCoeffs[2 * i] = TCoeffs[i];
		NumCoeffs[2 * i + 1] = 0.0;
	}
	NumCoeffs[2 * FilterOrder] = TCoeffs[FilterOrder];

	double cp[2];
	double Bw, Wn;
	cp[0] = 2 * 2.0*tan(M_PI * Lcutoff / 2.0);
	cp[1] = 2 * 2.0*tan(M_PI * Ucutoff / 2.0);

	Bw = cp[1] - cp[0];
	//center frequency
	Wn = sqrt(cp[0] * cp[1]);
	Wn = 2 * atan2(Wn, 4);
	double kern;
	const std::complex<double> result = std::complex<double>(-1, 0);

	for (int k = 0; k< FilterOrder * 2 + 1; k++)
	{
		NormalizedKernel[k] = std::exp(-sqrt(result)*Wn*Numbers[k]);
	}
	double b = 0;
	double den = 0;
	for (int d = 0; d < FilterOrder * 2 + 1; d++)
	{
		b += real(NormalizedKernel[d] * NumCoeffs[d]);
		den += real(NormalizedKernel[d] * DenC[d]);
	}
	for (int c = 0; c < FilterOrder * 2 + 1; c++)
	{
		NumCoeffs[c] = (NumCoeffs[c] * den) / b;
	}

	for (int i = NumCoeffs.size() - 1; i > FilterOrder * 2 + 1; i--)
		NumCoeffs.pop_back();

	return NumCoeffs;
}

std::vector<double> processing::CVSA::ComputeHP(int FilterOrder)
{
	std::vector<double> NumCoeffs;
	int i;

	NumCoeffs = this->ComputeLP(FilterOrder);

	for (i = 0; i <= FilterOrder; ++i)
		if (i % 2) NumCoeffs[i] = -NumCoeffs[i];

	return NumCoeffs;
}
std::vector<double> processing::CVSA::ComputeLP(int FilterOrder){
	std::vector<double> NumCoeffs(FilterOrder + 1);
	int m;
	int i;

	NumCoeffs[0] = 1;
	NumCoeffs[1] = FilterOrder;
	m = FilterOrder / 2;
	for (i = 2; i <= m; ++i)
	{
		NumCoeffs[i] = (double)(FilterOrder - i + 1)*NumCoeffs[i - 1] / i;
		NumCoeffs[FilterOrder - i] = NumCoeffs[i];
	}
	NumCoeffs[FilterOrder - 1] = FilterOrder;
	NumCoeffs[FilterOrder] = 1;

	return NumCoeffs;
}

std::vector<double> processing::CVSA::ComputeDenCoeffs(int FilterOrder, double Lcutoff, double Ucutoff){
	int k;            // loop variables
	double theta;     // PI * (Ucutoff - Lcutoff) / 2.0
	double cp;        // cosine of phi
	double st;        // sine of theta
	double ct;        // cosine of theta
	double s2t;       // sine of 2*theta
	double c2t;       // cosine 0f 2*theta
	std::vector<double> RCoeffs(2 * FilterOrder);     // z^-2 coefficients 
	std::vector<double> TCoeffs(2 * FilterOrder);     // z^-1 coefficients
	std::vector<double> DenomCoeffs;     // dk coefficients
	double PoleAngle;      // pole angle
	double SinPoleAngle;     // sine of pole angle
	double CosPoleAngle;     // cosine of pole angle
	double a;         // workspace variables

	cp = cos(M_PI * (Ucutoff + Lcutoff) / 2.0);
	theta = M_PI * (Ucutoff - Lcutoff) / 2.0;
	st = sin(theta);
	ct = cos(theta);
	s2t = 2.0*st*ct;        // sine of 2*theta
	c2t = 2.0*ct*ct - 1.0;  // cosine of 2*theta

    

	for (k = 0; k < FilterOrder; ++k)
	{
		PoleAngle = M_PI * (double)(2 * k + 1) / (double)(2 * FilterOrder);
		SinPoleAngle = sin(PoleAngle);
		CosPoleAngle = cos(PoleAngle);
		a = 1.0 + s2t*SinPoleAngle;
		RCoeffs[2 * k] = c2t / a;
		RCoeffs[2 * k + 1] = s2t*CosPoleAngle / a;
		TCoeffs[2 * k] = -2.0*cp*(ct + st*SinPoleAngle) / a;
		TCoeffs[2 * k + 1] = -2.0*cp*st*CosPoleAngle / a;

	}

	DenomCoeffs = this->TrinomialMultiply(FilterOrder, TCoeffs, RCoeffs);

	DenomCoeffs[1] = DenomCoeffs[0];
	DenomCoeffs[0] = 1.0;
	for (k = 3; k <= 2 * FilterOrder; ++k)
		DenomCoeffs[k] = DenomCoeffs[2 * k - 2];

	for (int i = DenomCoeffs.size() - 1; i > FilterOrder * 2 + 1; i--)
		DenomCoeffs.pop_back();

	return DenomCoeffs;
}

std::vector<double> processing::CVSA::TrinomialMultiply(int FilterOrder, std::vector<double> b, std::vector<double> c){
	int i, j;
	std::vector<double> RetVal(4 * FilterOrder);

	RetVal[2] = c[0];
	RetVal[3] = c[1];
	RetVal[0] = b[0];
	RetVal[1] = b[1];

	for (i = 1; i < FilterOrder; ++i)
	{
		RetVal[2 * (2 * i + 1)] += c[2 * i] * RetVal[2 * (2 * i - 1)] - c[2 * i + 1] * RetVal[2 * (2 * i - 1) + 1];
		RetVal[2 * (2 * i + 1) + 1] += c[2 * i] * RetVal[2 * (2 * i - 1) + 1] + c[2 * i + 1] * RetVal[2 * (2 * i - 1)];

		for (j = 2 * i; j > 1; --j)
		{
			RetVal[2 * j] += b[2 * i] * RetVal[2 * (j - 1)] - b[2 * i + 1] * RetVal[2 * (j - 1) + 1] +
				c[2 * i] * RetVal[2 * (j - 2)] - c[2 * i + 1] * RetVal[2 * (j - 2) + 1];
			RetVal[2 * j + 1] += b[2 * i] * RetVal[2 * (j - 1) + 1] + b[2 * i + 1] * RetVal[2 * (j - 1)] +
				c[2 * i] * RetVal[2 * (j - 2) + 1] + c[2 * i + 1] * RetVal[2 * (j - 2)];
		}

		RetVal[2] += b[2 * i] * RetVal[0] - b[2 * i + 1] * RetVal[1] + c[2 * i];
		RetVal[3] += b[2 * i] * RetVal[1] + b[2 * i + 1] * RetVal[0] + c[2 * i + 1];
		RetVal[0] += b[2 * i];
		RetVal[1] += b[2 * i + 1];
	}

	return RetVal;
}


/*
void processing::CVSA::butterworthBandpass(int order, double lowcut, double highcut, double fs, std::vector<double>& a, std::vector<double>& b) {
    double nyquist = 0.5 * fs;
    double low = lowcut / nyquist;
    double high = highcut / nyquist;

    std::vector<double> a0(order + 1), a1(order + 1), a2(order + 1);
    std::vector<double> b0(order + 1), b1(order + 1), b2(order + 1);

    double t = tan(M_PI * (high - low) / 2.0);
    double r = sqrt(t * t + 1.0) + t;

    for (int i = 0; i <= order; i++) {
        double theta = M_PI * (2.0 * i + 1.0) / (2.0 * order); 
        double sigma = -cos(theta);
        double omega = sin(theta);

        a0[i] = 1.0;
        a1[i] = sigma * r;
        a2[i] = (sigma * sigma + omega * omega) * r * r;
        b0[i] = 1.0;
        b1[i] = 2.0 * sigma;
        b2[i] = sigma * sigma + omega * omega;
    }

    // Combine stages to form final coefficients
    for (int i = 0; i <= order; i++) {
        a.push_back(a0[i]);
        a.push_back(a1[i]);
        a.push_back(a2[i]);
        b.push_back(b0[i]);
        b.push_back(b1[i]);
        b.push_back(b2[i]);
    }

    std::string a_s = "";
    std::string b_s = "";
    for(int i = 0; i < a.size(); i++){
        a_s += " " + std::to_string(a[i]);
        b_s += " " + std::to_string(b[i]);
    }
    ROS_INFO("a: %s ", a_s.c_str());
    ROS_INFO("b: %s ", b_s.c_str());
}

Eigen::MatrixXd processing::CVSA::applyFilterToMatrix(const Eigen::VectorXd& b, const Eigen::VectorXd& a, const Eigen::MatrixXd& data) {
    Eigen::MatrixXd filteredData(data.rows(), data.cols());
    for (int row = 0; row < data.rows(); ++row) {
        filteredData.row(row) = applyFilter(b, a, data.row(row).transpose()).transpose();
    }
    return filteredData;
}

Eigen::VectorXd processing::CVSA::applyFilter(const Eigen::VectorXd& b, const Eigen::VectorXd& a, const Eigen::VectorXd& data) {
    Eigen::VectorXd filteredData(data.size());
    filteredData.setZero();
    
    for (int i = 0; i < data.size(); ++i) {
        filteredData[i] = data[i] * b[0];
        for (int j = 1; j < b.size(); ++j) {
            if (i >= j) {
                filteredData[i] += data[i - j] * b[j] - filteredData[i - j] * a[j];
            }
        }
    }
    return filteredData;
}
*/