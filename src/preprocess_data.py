import os
import requests
import zipfile
import xml.etree.ElementTree as ET


def download_weekly_patents(year, month, day):
    """
    Download weekly patent files from the USPTO website based on a specific date.

    Parameters:
    year (int): The year of the patent.
    month (int): The month of the patent.
    day (int): The day of the patent.

    Returns:
    bool: True if the download is successful, False otherwise.
    """

    # file_path_check = os.path.join(
    #     os.getcwd(),
    #     "data",
    #     "ipa" + str(year)[2:] + f"{month:02d}" + f"{day:02d}" + ".xml",
    # )

    # if os.path.exists(file_path_check):
    #     print(f"File {file_path_check} already exists. Skipping download.")
    #     return True

    # Check if the "data" folder exists and create one if it doesn't
    data_folder = os.path.join(os.getcwd(), "data")
    if not os.path.exists(data_folder):
        print("Data folder not found. Creating a new 'data' folder.")
        os.makedirs(data_folder)

    directory = os.path.join(
        os.getcwd(), "data", "ipa" + str(year)[2:] + f"{month:02d}" + f"{day:02d}"
    )

    if os.path.exists(directory):
        print(f"File {directory} already exists. Skipping download.")
        return True

    print("Building the URL...")
    base_url = "https://bulkdata.uspto.gov/data/patent/application/redbook/fulltext"
    file_url = (
        base_url
        + "/"
        + str(year)
        + "/ipa"
        + str(year)[2:]
        + f"{month:02d}"
        + f"{day:02d}"
        + ".zip"
    )

    print(f"URL constructed: {file_url}")
    r = requests.get(file_url, stream=True)

    print("Requesting the file...")
    if r.status_code == 200:
        print("File retrieved successfully. Starting download...")
        local_path = os.path.join(os.getcwd(), "data", "patents.zip")

        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("File downloaded successfully. Starting extraction...")
        with zipfile.ZipFile(local_path, "r") as zip_ref:
            zip_ref.extractall(os.path.join(os.getcwd(), "data"))

        print("File extracted successfully.")
        # Deleting the ZIP file after extraction
        os.remove(local_path)
        print(f"ZIP file {local_path} deleted after extraction.")

        return True
    else:
        print(
            "File could not be downloaded. Please make sure the year, month, and day are correct."
        )
        return False


def extract_patents(year, month, day):
    """
    This function reads a patent file in XML format, splits it into individual patents and
    saves each patent as a separate XML file in a directory named 'data'.

    Parameters:
    year (int): The year of the patent file to process.
    month (int): The month of the patent file to process.
    day (int): The day of the patent file to process.

    Returns:
    None

    The function creates a separate XML file for each patent and stores these files in
    a directory. The directory is named based on the year, month and day provided.
    If the directory does not exist, the function creates it. The function also prints
    the total number of patents found.

    """

    directory = os.path.join(
        os.getcwd(), "data", "ipa" + str(year)[2:] + f"{month:02d}" + f"{day:02d}"
    )

    if os.path.exists(directory):
        print(f"File {directory} already exists. Skipping extract.")
        return True

    print("Locating the patent file...")
    file_path = os.path.join(
        os.getcwd(),
        "data",
        "ipa" + str(year)[2:] + f"{month:02d}" + f"{day:02d}" + ".xml",
    )

    print("Reading the patent file...")
    with open(file_path, "r") as f:
        contents = f.read()

    print("Splitting the patent file into individual patents...")
    temp = contents.split("</us-patent-application>")
    patents = [s.replace("\n", "") + "</us-patent-application>" for s in temp]
    print(f"Total patents found: {len(patents)}")

    print("Creating directory to store individual patents...")
    directory = os.path.join(
        os.getcwd(), "data", "ipa" + str(year)[2:] + f"{month:02d}" + f"{day:02d}"
    )
    if not os.path.exists(directory):
        os.mkdir(directory)
    print("Writing individual patents to separate XML files...")
    for i, xml_str in enumerate(patents):
        # Create a filename based on the index of the string
        filename = os.path.join(directory, f"{i}.xml")

        # Write the string to a new file
        with open(filename, "w") as f:
            f.write(xml_str)

    print("Patent extraction complete.")

    # Deleting the main XML file after extraction
    os.remove(file_path)
    print(f"Main XML file {file_path} deleted after extraction.")

    return True


def get_full_text(element):
    """
    Recursively parse XML elements and retrieve the full text from the XML tree.

    Parameters:
        element (xml.etree.ElementTree.Element): The root XML element to start parsing.

    Returns:
        list: A list of strings containing the full text from the XML element and its children.
    """

    text = []
    if element.text is not None and element.text.strip():
        text.append(element.text.strip())
    for child in element:
        text.extend(get_full_text(child))
        if child.tail is not None and child.tail.strip():
            text.append(child.tail.strip())
    return text


def compare_txt_vs_xml_files(year, month, day):
    """
    Compare the number of TXT files vs XML files in the specified directory for a given date.

    Parameters:
        year (int): The year of the patents to compare.
        month (int): The month of the patents to compare.
        day (int): The day of the patents to compare.

    Returns:
        None

    This function calculates the number of XML and TXT files in the specified directory
    corresponding to the given date. It then computes the percentage of success by
    comparing the difference between the number of TXT files and XML files against the total
    number of files. The result is printed to the console.
    """

    data_directory = os.path.join(
        os.getcwd(), "data", "ipa" + str(year)[2:] + f"{month:02d}" + f"{day:02d}"
    )
    xml_files = [file for file in os.listdir(data_directory) if file.endswith(".xml")]
    txt_files = [file for file in os.listdir(data_directory) if file.endswith(".txt")]

    num_xml_files = len(xml_files)
    num_txt_files = len(txt_files)

    percentage_success = ((num_txt_files) / (num_txt_files + num_xml_files)) * 100

    print(f"Number of XML files: {num_xml_files}")
    print(f"Number of TXT files: {num_txt_files}")
    print(f"Percentage of success: {percentage_success:.2f}%")


def parse_and_save_patents(year, month, day):
    """
    Download weekly patent files from the USPTO website for a specific date, extract individual
    patents from the downloaded file, parse each patent's content, and save the information
    as separate text files.

    Parameters:
        year (int): The year of the weekly patents to download and process.
        month (int): The month of the weekly patents to download and process.
        day (int): The day of the weekly patents to download and process.

    Returns:
        list: A list of strings containing the names of saved patent text files.

    This function first downloads the weekly patent file, then extracts the individual patents,
    and finally parses each patent's content to retrieve patent_id, file_id, and full text.
    It saves the extracted information for each patent as separate text files in a directory
    named 'data', with the name of each file being the corresponding 'file_id'.
    The function returns a list of strings containing the names of all the saved patent text files.
    """

    print("### Downloading weekly patent files...")
    download_success = download_weekly_patents(year, month, day)
    if not download_success:
        print("Failed to download the weekly patents.")
        return

    print("### Extracting individual patents...")
    extraction_success = extract_patents(year, month, day)
    if not extraction_success:
        print("Failed to extract the individual patents.")
        return

    data_directory = os.path.join(
        os.getcwd(), "data", "ipa" + str(year)[2:] + f"{month:02d}" + f"{day:02d}"
    )
    xml_files = [file for file in os.listdir(data_directory) if file.endswith(".xml")]

    saved_patent_names = []

    for xml_file in xml_files:
        file_path = os.path.join(data_directory, xml_file)

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            patent_id = root.find(
                ".//publication-reference/document-id/doc-number"
            ).text
            file_id = root.attrib["file"]

            ipcr_classifications = root.findall(".//classification-ipcr")

            if any(ipcr.find("./section").text == "C" for ipcr in ipcr_classifications):
                description_element = root.find(".//description")
                description_text = get_full_text(description_element)
                description_string = " ".join(description_text)

                output_file_path = os.path.join(data_directory, f"{file_id}.txt")
                with open(output_file_path, "w") as f:
                    f.write(
                        f"-patent_id: {patent_id} -file_id: {file_id} -full text: {description_string}"
                    )

                print(
                    f"Information extracted from {xml_file} and saved in {output_file_path}"
                )
                saved_patent_names.append(f"{file_id}.txt")

            else:
                print(
                    f"XML file {file_path} does not belong to section 'C'. Removing this file."
                )

            os.remove(file_path)

        except ET.ParseError as e:
            print(f"Error while parsing XML file: {file_path}. Skipping this file.")
            print(f"Error message: {e}")

    print("Patent parsing and saving complete.")

    compare_txt_vs_xml_files(year, month, day)

    return saved_patent_names
