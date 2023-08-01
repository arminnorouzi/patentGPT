# USPTO patent analysis

This repository hosts a prototype for patent analysis with a particular focus on extracting specific technical measurements and their associated values

## Quick Start

1. Clone this repository.
2. Create a new environment
3. Install the required Python packages by running `pip install -r requirements.txt` in your terminal.
4. Run the Jupyter notebook `patent_analysis.ipynb`.

## System Design

Below is an image that illustrates the main design and workflow of the patent analysis system

![System Design](images/system.png)

The project involves the following steps:

1. Downloading a ZIP archive that contains granted patent full-text data (without images) from https://bulkdata.uspto.gov/.
2. Reading the contained patents from XML files and extracting individual XML file, parsing it to text file and saving it in data.
3. Implementing an approach based on a large language model (LLM) to extract measurements from the patents using vector store Chroma. The measurements are returned in a structured format (such as JSON).
4. Analyzing the model's results, and discussing the results/output of the models, including typical errors and points for improvement.
5. Evaluating output based on human annotator and logging and monitoring the accuracy of the system over time.

## Directory Structure

- root/
  - data/
  - output/
  - src/
  - requirements.txt
  - LICENSE
  - README.md
  - patent_analysis.ipynb
  - images
  - presentation

## Requirements

- Python 3.10+
- See requirements.txt for Python packages and versions.

## License

This project is licensed under the terms of the MIT license. See [LICENSE](LICENSE) file.

## Additional Resources

For further questions or if you encounter any problems, please do not hesitate to open an issue.

- [My Website](https://arminnorouzi.github.io/) - More details about me and my projects.
- [LinkedIn](https://www.linkedin.com/in/arminnorouzi/) - See the post related to machine learning and software development on LinkedIn.
- [Medium](https://arminnorouzi.medium.com/) - See my post related to ML/AI, algorithms, and system design.
