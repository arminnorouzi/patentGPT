import os
import requests
import zipfile
import xml.etree.ElementTree as ET
import pickle


def download_weekly_patents(year, month, day, logging):
    """
    Download weekly patent files from the USPTO website based on a specific date.

    Parameters:
    year (int): The year of the patent.
    month (int): The month of the patent.
    day (int): The day of the patent.
    logging (bool): The boolean to print logs

    Returns:
    bool: True if the download is successful, False otherwise.
    """

    # Check if the "data" folder exists and create one if it doesn't
    data_folder = os.path.join(os.getcwd(), "data")
    if not os.path.exists(data_folder):
        if logging:
            print("Data folder not found. Creating a new 'data' folder.")
        os.makedirs(data_folder)

    directory = os.path.join(
        os.getcwd(), "data", "ipa" + str(year)[2:] + f"{month:02d}" + f"{day:02d}"
    )

    if os.path.exists(directory):
        print(f"File {directory} already exists. Skipping download.")
        return True

    if logging:
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

    if logging:
        print(f"URL constructed: {file_url}")
    r = requests.get(file_url, stream=True)

    if logging:
        print("Requesting the file...")
    if r.status_code == 200:
        if logging:
            print("File retrieved successfully. Starting download...")
        local_path = os.path.join(os.getcwd(), "data", "patents.zip")

        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        if logging:
            print("File downloaded successfully. Starting extraction...")
        with zipfile.ZipFile(local_path, "r") as zip_ref:
            zip_ref.extractall(os.path.join(os.getcwd(), "data"))

        if logging:
            print("File extracted successfully.")
        # Deleting the ZIP file after extraction
        os.remove(local_path)
        if logging:
            print(f"ZIP file {local_path} deleted after extraction.")

        return True
    else:
        print(
            "File could not be downloaded. Please make sure the year, month, and day are correct."
        )
        return False


def extract_patents(year, month, day, logging):
    """
    This function reads a patent file in XML format, splits it into individual patents, parse each
    XML file and saves each patent as a separate txt file in a directory named 'data'.

    Parameters:
    year (int): The year of the patent file to process.
    month (int): The month of the patent file to process.
    day (int): The day of the patent file to process.
    logging (bool): The boolean to print logs

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
    saved_patent_names_path = os.path.join(directory, 'saved_patent_names.pkl')


    if os.path.exists(directory):
        print(f"File {directory} already exists. Skipping extract.")
        
        # Load saved_patent_names from file
        with open(saved_patent_names_path, 'rb') as f:
            saved_patent_names = pickle.load(f)
            
        return saved_patent_names
    else:
        os.mkdir(directory)

    if logging:
        print("Locating the patent file...")
    file_path = os.path.join(
        os.getcwd(),
        "data",
        "ipa" + str(year)[2:] + f"{month:02d}" + f"{day:02d}" + ".xml",
    )

    if logging:
        print("Reading the patent file...")
    with open(file_path, "r") as f:
        contents = f.read()

    if logging:
        print("Splitting the XMl file into individual XMLs...")
    temp = contents.split('<?xml version="1.0" encoding="UTF-8"?>')
    allXmls = [
        '<?xml version="1.0" encoding="UTF-8"?>' + s.replace("\n", "") for s in temp
    ]

    # saving only the XMLs that contain a patent
    patents = []
    for xml_string in allXmls:
        start_index = xml_string.find("<!DOCTYPE")
        end_index = xml_string.find(">", start_index)

        if start_index != -1 and end_index != -1:
            doctype_declaration = xml_string[start_index : end_index + 1]
            # Extract only the name of the DOCTYPE
            doctype_name = doctype_declaration.split()[1]
            if doctype_name == "us-patent-application":
                patents.append(xml_string)

    if logging:
        print(f"Total patents found: {len(patents)}")
        print("Writing individual patents to separate txt files...")
    
    saved_patent_names = []
    for patent in patents:
        try:
            root = ET.fromstring(patent)

            patent_id = root.find(
                ".//publication-reference/document-id/doc-number"
            ).text
            file_id = root.attrib["file"]

            ipcr_classifications = root.findall(".//classification-ipcr")

            if any(ipcr.find("./section").text == "C" for ipcr in ipcr_classifications):
                description_element = root.find(".//description")
                description_text = get_full_text(description_element)
                description_string = " ".join(description_text)

                output_file_path = os.path.join(directory, f"{file_id}.txt")
                with open(output_file_path, "w") as f:
                    f.write(description_string)
                saved_patent_names.append(f"{file_id}.txt")

            elif logging:
                print(
                    f"Patent {patent_id} does not belong to section 'C'. Skipping this patent."
                )
        except ET.ParseError as e:
            print(f"Error while parsing patent: {patent_id}. Skipping this patent.")
            print(f"Error message: {e}")

    # Save saved_patent_names to file
    with open(saved_patent_names_path, 'wb') as f:
        pickle.dump(saved_patent_names, f)

    if logging:
        print("Patent extraction complete.")

    # Deleting the main XML file after extraction
    os.remove(file_path)

    if logging:
        print(f"Main XML file {file_path} deleted after extraction.")
    return saved_patent_names


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


def parse_and_save_patents(year, month, day, logging=False):
    """
    Download weekly patent files from the USPTO website for a specific date, extract individual
    patents from the downloaded file, parse each patent's content, and save the information
    as separate text files.

    Parameters:
        year (int): The year of the weekly patents to download and process.
        month (int): The month of the weekly patents to download and process.
        day (int): The day of the weekly patents to download and process.
        logging (bool): The boolean to print logs

    Returns:
        list: A list of strings containing the names of saved patent text files.

    This function first downloads the weekly patent file, then extracts the individual patents,
    and finally parses each patent's content to retrieve patent_id, file_id, and full text.
    It saves the extracted information for each patent as separate text files in a directory
    named 'data', with the name of each file being the corresponding 'file_id'.
    The function returns a list of strings containing the names of all the saved patent text files.
    """

    if logging:
        print("### Downloading weekly patent files...")
    download_success = download_weekly_patents(year, month, day, logging)
    if not download_success:
        print("Failed to download the weekly patents.")
        return

    if logging:
        print("### Extracting individual patents...")
    saved_patent_names = extract_patents(year, month, day, logging)

    return saved_patent_names
