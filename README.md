## CVSA Processing Node

This package is responsible for processing raw EEG data, calculating signal power across different frequency bands, and publishing the results.

The code is designed to run as a ROS node but can also be integrated as a C++ class, though it maintains dependencies on `rosneuro` libraries.

---

### 1. Input

* **Topic:** `/neurodata`
* **Data:** Raw EEG signal, received in chunks of 32 samples at a 512 Hz sampling rate.

---

### 2. Processing Workflow

1.  **Ring Buffer:** The node uses a `rosneuro/ringbuffer` (configured via `ringbuffer.yaml`) to accumulate 512 samples. This buffer structure effectively manages the data flow, allowing the 512 Hz input stream (in chunks of 32) to be processed and downsampled to 16 Hz.

2.  **Filtering:** Once the buffer is filled, the 512 samples undergo processing:
    * **Band-pass Filtering:** The data is filtered using one or more band-pass filters.
    * **Configuration:** The specific frequency bands (e.g., `[delta, theta, alpha]`) are specified via the `filters_band` parameter in the launch file.
    * **Filter Order:** The filters default to order 4, which can also be modified via the launch file.

3.  **Power Calculation:**
    * The **Hilbert Transform** is applied to the filtered data.
    * The **instantaneous power** is calculated from the Hilbert-transformed signal.
    * The **mean power** is then computed across the buffer.

---

### 3. Output

* **Topic:** `/cvsa/eeg_power`
* **Message Type:** This package defines a **custom message type** for this topic.

The custom message contains the following fields:
* `n_channels`: The number of EEG channels.
* `n_bands`: The number of frequency bands processed.
* `eeg_code`: A unique identifier for the EEG signal, matching the code from the corresponding frame on `/neurodata`.
* `bands`: A list of strings (e.g., `['delta', 'theta']`) specifying the bands that were processed.
* `data`: The calculated mean power, structured as a flattened matrix: `[channels x bands]`.

---

### 4. Configuration

* **`cfg/ringbuffer.yaml`**: This file is required to configure the `rosneuro/ringbuffer` used by the node.

---

### 5. Testing

The `test` directory contains two primary validation tests, both of which use `rawdata.csv` as a common input file for comparison against a MATLAB benchmark implementation.

1.  **Class Test:** This test validates the C++ processing class in isolation. It confirms that the C++ implementation of the filtering and power calculation logic produces results identical to the MATLAB implementation.

2.  **Node Test:** This test validates the full ROS node. By using publishers and subscribers, it confirms not only that the processing logic is correct (comparing the final output to MATLAB) but also that the ROS communication (data subscription and publication) functions as expected.