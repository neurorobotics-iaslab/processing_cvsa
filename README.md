## CVSA Processing Node

This package is responsible for processing raw EEG data, calculating signal power across different frequency bands, and publishing the results.

The code is designed to run as a ROS node but can also be integrated as a C++ class, though it maintains dependencies on `rosneuro` libraries.

---

### 1. Input

* **Topic:** `/neurodata`
* **Data:** Raw EEG signal. The data ingestion is fully parametric: it receives dynamic chunks of samples depending on your `chunkSize` and `samplerate` configurations. Specifically, the `samplerate` must match the hardware acquisition parameter, while the `chunkSize` depends entirely on the desired ROS loop framerate (e.g., `samplerate / framerate = chunkSize`).

---

### 2. Processing Workflow

1.  **Format Input Data:** The node evaluates `run_mode` (`online` or `offline`) and `signal_type` (`eeg` or `eeg_eog`) to reconstruct the incoming data matrix dynamically, gracefully handling different data structures (e.g., LSL playback exg placements).
2.  **CAR Spatial Filter:** The loaded data vector is first passed through a **Common Average Reference (CAR)** filter, which mitigates noise evenly distributed across channels (configured via a dedicated YAML file).
3.  **Ring Buffer:** The node utilizes a `rosneuro/ringbuffer`. The capacity of this buffer is also strictly parametric and configured externally. Typically scaled to contain 1 second of data, it naturally adapts to the configured sampling rate (e.g., buffering 512 samples for a 512 Hz rate, or 250 samples for a 250 Hz rate), managing high-frequency chunk streams gracefully.
4.  **Frequency Filtering:** 
    * **Band-pass Filtering:** The accumulated data is filtered through one or multiple independent IIR Butterworth band-pass filters in parallel.
    * **Configuration:** Frequency bands (e.g., `8.0-14.0`, `18.0-24.0`) are provided via the `filters_band` parameter.
    * **Order:** Filter order defaults to 4.
5.  **Hann Windowing (Optional):** If the `do_hann` parameter is `true`, a Hann window is applied to the buffered chunk to gracefully reduce spectral leakage at the boundaries before transformation.
6.  **Power Calculation:**
    * A Fast Fourier Transform (FFTW) calculates the **analytic signal** by eliminating negative frequencies and doubling the positive spectrum.
    * The **instantaneous power** is derived (squared absolute value).
    * Finally, the **mean power** is computed across the buffer window to extract the discrete feature.

---

### 3. Output

* **Topic:** Default is `/eeg_power`, but it can be dynamically overridden via the `topic_to_pub` ros param.
* **Message Type:** This package defines a **custom message type** for this topic.

The custom message contains the following fields:
* `n_channels`: The number of EEG channels.
* `n_bands`: The number of frequency bands processed.
* `eeg_code`: A unique identifier for the EEG signal, matching the code from the corresponding frame on `/neurodata`.
* `bands`: A list of strings (e.g., `['delta', 'theta']`) specifying the bands that were processed.
* `data`: The calculated mean power, structured as a flattened matrix: `[channels x bands]`.

---

### 4. Configuration

To properly instantiate the node, the launch file must supply several node-level variables and load necessary YAML mappings:

**Node Parameters:**
* `nchannels`: Total integer number of main channels acquired.
* `samplerate`: The hardware sampling frequency used during data acquisition (e.g., 250 or 512 Hz).
* `chunkSize`: Sample chunk length. This depends directly on the ROS framerate you wish to maintain (e.g., `chunkSize = samplerate / framerate`).
* `signal_type`: The source structural type (e.g., `eeg` or `eeg_eog`).
* `run_mode`: Define if the stream is `online` (live acquisition) or `offline` (simulated playback).
* `filter_order` / `filters_band`: Core filtering characteristics.
* `do_hann`: `bool`. Activates the Hann windowing on the ring buffer prior to power transformation.

**Required YAML files:**
* **`cfg/ringbuffer.yaml`**: Necessary to configure the size and scope of the `rosneuro/ringbuffer`.
* **`cfg/car.yaml`**: Configures the spatial `CarCfg` namespace (defining which channels to evaluate or exclude during spatial referencing).

---

### 5. Testing

The `test` directory contains two primary validation tests, both of which use `rawdata.csv` as a common input file for comparison against a MATLAB benchmark implementation.

1.  **Class Test:** This test validates the C++ processing class in isolation. It confirms that the C++ implementation of the filtering and power calculation logic produces results identical to the MATLAB implementation.

2.  **Node Test:** This test validates the full ROS node. By using publishers and subscribers, it confirms not only that the processing logic is correct (comparing the final output to MATLAB) but also that the ROS communication (data subscription and publication) functions as expected.