# fiberphotometry
This code is primarily based on the following repository:
https://github.com/ThomasAkam/photometry_preprocessing

🔧 Main Modifications
Added preprocessing code to handle data acquired using time division
Added methods for analyzing behavioral data
Implemented functions to align and analyze neural and behavioral data on a shared timeline
Introduced additional analytical metrics

⚙️ Before You Start
When analyzing neural data, please modify the necessary variables in neural_analysis.py according to your experimental conditions:
test_zero_time: indicates the start time of the test (e.g., 0)
test_time: indicates the end time of the test (e.g., 300)
The behavioral data analysis assumes the use of Melquest experimental equipment.
