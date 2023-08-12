import nltk

nltk.download("all", quiet=True)
from datetime import datetime
import random
import json
from . import preprocess_data
from . import qaagent


PROMPT = """
Task: Carefully review the given patent text and extract as much physical measurements information such as length/distance, mass/weight, time, temperature, Volume, area, speed, pressure, energy, power, electric current 
and voltage, frequency, force, acceleration, density, resistivity, magnetic field strength, and luminous intensity as much as possible. 
We are particularly interested in physical measurements including substance that was measured, Value of the measurement, and Unit of the measurement, and measurement type mentioned in the text. 

For each measurement, please provide the following details:
- The substance that was measured. (substance)
- The specific value or range that was measured. (Measured Value)
- The unit of the measurement, if provided. (Unit)
- The type of measurement being conducted (e.g., diameter, size, etc.) 


Format your response in a structured JSON-like format, as follows:

{"Content": [
    {
      "Measurement_substance": "substance",
      "Measured_value": "value",
      "Measured_unit": "unit",
      "measurement_type": "type"
    },
    // ... additional measurements, if present
  ]
}

If multiple measurements are present in the text, each should be listed as a separate object within the "Content" array.

Example: If the text includes the sentence, "The resulting BaCO3 had a crystallite size of between about 20 and 40 nm", the output should be:

{"Content": [
    {
      "Measurement_substance": "BaCO3",
      "Measured_value": "between about 20 and 40",
      "Measured_unit": "nm",
      "measurement_type": "crystallite size"
    }
  ]
}

Try to provide as complete and accurate information as possible. Print only the formatted JSON response.
"""


def main():
    """
    Main function to:
    - Authenticate with OpenAI
    - Receive and parse date input from the user
    - Extract and print year, month, day
    - Preprocess patent data
    - Analyze selected patents using GPT-3.5 Turbo
    - Print results including cost and optionally output
    """
    print("Starting the patent analysis process...")
    # Step 1: Input the date from the user
    user_date_input = input("Enter a date in the format 'YYYY-MM-DD': ")

    # Step 2: Parse the input date into a datetime object
    try:
        input_date = datetime.strptime(user_date_input, "%Y-%m-%d")
    except ValueError:
        print(
            "Invalid date format. Please enter a valid date in the format 'YYYY-MM-DD'."
        )
        return

    # Step 3: Extract date components
    year = input_date.year
    month = input_date.month
    day = input_date.day

    print("Year:", year)
    print("Month:", month)
    print("Day:", day)

    # Step 4: Get random patents number from user
    num_patents_to_analyze = int(
        input("Enter the number of patents you want to analyze: ")
    )

    logging_choice = input("Do you want to log the results? (yes/no): ").strip().lower()
    logging_enabled = logging_choice == "yes"

    model_choice = input(
        "Select a model for analysis: 1. gpt-3.5-turbo 2. gpt-4"
    ).strip()

    if model_choice == "1":
        model_name = "gpt-3.5-turbo"
    elif model_choice == "2":
        model_name = "gpt-4"
    else:
        print("Invalid choice, defaulting to gpt-3.5-turbo.")
        model_name = "gpt-3.5-turbo"

    print("Processing patents...")
    # Step 5: Parse and save patents
    saved_patent_names = preprocess_data.parse_and_save_patents(year, month, day, False)

    # Step 6: Select random patents and analyze
    random_patents = random.sample(saved_patent_names, num_patents_to_analyze)

    gpt_3_results = {}
    total_cost_gpt3 = 0

    # Step 7: Process patents with GPT-3.5 Turbo
    for i in range(len(random_patents)):
        cost, output = qaagent.call_QA_to_json(
            PROMPT, year, month, day, random_patents, i, logging_enabled, model_name
        )

        total_cost_gpt3 += cost

    average_cost_gpt3 = total_cost_gpt3 / num_patents_to_analyze

    print("Patent analysis process completed successfully.")
    # Step 8: Print results
    print("\nResults for GPT-3.5 Turbo:")
    print("Number of patents analyzed:", num_patents_to_analyze)
    print("Total cost for analyzing all patents:", total_cost_gpt3)
    print("Average cost per patent:", average_cost_gpt3)
