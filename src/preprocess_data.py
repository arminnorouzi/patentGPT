import os
import requests
import zipfile


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

    file_path_check = os.path.join(
        os.getcwd(),
        "data",
        "ipa" + str(year)[2:] + f"{month:02d}" + f"{day:02d}" + ".xml",
    )

    if os.path.exists(file_path_check):
        print(f"File {file_path_check} already exists. Skipping download.")
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
