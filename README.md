# PatentGPT

This repository hosts a prototype for patent analysis with a particular focus on extracting specific technical measurements and their associated values

[![PyPI version](https://badge.fury.io/py/patentgpt-extract.svg)](https://pypi.org/project/patentgpt-extract/) [![License](https://img.shields.io/github/license/arminnorouzi/patentGPT)](https://github.com/arminnorouzi/patentGPT/blob/main/LICENSE)

## Quick Start

To run this package in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arminnorouzi/patentGPT/blob/main/quick_start.ipynb)

## Installation

Step 1. Install `patentgpt-extract`, simply use `pip`:

`from patentgpt.main import main`

Step 2. Authentication: Before using the package, you need to authenticate with OpenAI. To do this:

```
import os
from getpass import getpass

token = getpass("Enter your OpenAI token: ()")
os.environ["OPENAI_API_KEY"] = str(token)
```

Step 3. Importing and Running:

```
from patentgpt.main import main

main()
```

Then answer these questions::

- Enter a date in the format 'YYYY-MM-DD': exampe is 2023-01-12
- Enter the number of patents you want to analyze: example is 5 (this randomly select 5 parsed patents)
- Do you want to log the results? (yes/no)
- Select a model for analysis: 1. gpt-3.5-turbo 2. gpt-4

## Quick Start using repository

1. Clone this repository.
2. Install the required Python packages by running `!pip install -r requirements.txt`.
3. Run the Jupyter notebook `quick_start.ipynb`.
4. Authenticate and add your OpenAI token. Then answer these questions::
   - Enter a date in the format 'YYYY-MM-DD': exampe is 2023-01-12
   - Enter the number of patents you want to analyze: example is 5 (this randomly select 5 parsed patents)
   - Do you want to log the results? (yes/no)
   - Select a model for analysis: 1. gpt-3.5-turbo 2. gpt-4
5. JSON results will be saved in the output folder.

## System Design

Below is an image that illustrates the main design and workflow of the patent analysis system

![System Design](https://github.com/arminnorouzi/patentGPT/blob/main/images/system.png?raw=true)

The project involves the following steps:

1. Downloading a ZIP archive that contains granted patent full-text data (without images) from https://bulkdata.uspto.gov/.
2. Reading the contained patents from XML files and extracting individual XML file, parsing it to text file and saving it in data.
3. Implementing an approach based on a large language model (LLM) to extract measurements from the patents using vector store Chroma. The measurements are returned in a structured format (such as JSON).

## Requirements

- Python 3.10+
- See requirements.txt for Python packages and versions.

## License

This project is licensed under the terms of the MIT license. See [LICENSE](LICENSE) file.

## Collaboration

We welcome contributions to patentgpt-extract! If you're interested in improving the package, adding features, or even fixing bugs, here's how you can get started:

1. Fork the Repository: Start by forking the repository. This creates your own personal copy of the entire project.

2. Clone Your Fork: Once you've forked the repo, clone your fork to your local machine to start making changes.

```
git clone https://github.com/arminnorouzi/patentGPT.git
```

3. Create a New Branch: Before making changes, create a new branch. This helps in segregating your changes and makes it easier to merge later.

```
git checkout -b new-feature-branch
```

Replace `new-feature-branch`` with a descriptive name for your changes.

4. Make Your Changes: Now, you can start making changes, adding new features, fixing bugs, or improving documentation.

5. Commit and Push: Once you're done, commit your changes and push them to your fork on GitHub.

```
git add .
git commit -m "feat or fix: Description of changes made"
git push origin new-feature-branch
```

6. Open a Pull Request: Go to your fork on GitHub and click the "New pull request" button. Ensure you're comparing the correct branches and then submit your pull request with a description of the changes you made.

## Additional Resources

For further questions or if you encounter any problems, please do not hesitate to open an issue.

- [My Website](https://arminnorouzi.github.io/) - More details about me and my projects.
- [LinkedIn](https://www.linkedin.com/in/arminnorouzi/) - See the post related to machine learning and software development on LinkedIn.
- [Medium](https://arminnorouzi.medium.com/) - See my posts related to ML/AI, algorithms, and system design.
