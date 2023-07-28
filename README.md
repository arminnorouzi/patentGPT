# USPTO patent analysis
This repository hosts a prototype for patent analysis with a particular focus on extracting specific technical measurements and their associated values

## Quick Start
1. Clone this repository.
2. Install the required Python packages by running `pip install -r requirements.txt` in your terminal.
3. Run the Jupyter notebook `notebooks/patent_analysis.ipynb`.

## Project Steps
The project involves the following steps:

1. Downloading a ZIP archive that contains granted patent full-text data (without images) from https://bulkdata.uspto.gov/.
2. Reading the contained patents from XML files and extracting individual XML file and saving it in data.
3. Implementing an approach based on a large language model (LLM) to extract measurements from the patents. The measurements are returned in a structured format (such as JSON).
4. Analyzing the model's results, and discussing the results/output of the models, including typical errors and points for improvement.


## Requirements
- Python 3.7+
- See requirements.txt for Python packages and versions.



License
This project is licensed under the terms of the MIT license. See [LICENSE](LICENSE) file.

For further questions or if you encounter any problems, please do not hesitate to open an issue.

