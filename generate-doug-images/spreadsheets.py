import os.path
import logging

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.external_account_authorized_user import Credentials as ExtCredentials

from typing import Any, Sequence, Literal

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", ]

# The ID and range of a sample spreadsheet.
# TODO: Move these to where they'll actually be used. Load SAMPLE_SPREADSHEET_ID from a file or envvar.
SAMPLE_SPREADSHEET_ID = "101l4E71e8m0uTOQmt8ZySwf9PPu0Ep51uJ8YresA_uQ"
SAMPLE_RANGE_NAME = "Sheet1!A2:C"

InputValueOption = Literal["RAW", "USER_ENTERED", "INPUT_VALUE_OPTION_UNSPECIFIED"]

def init_credentials() -> Credentials | ExtCredentials:
  """Shows basic usage of the Sheets API.
  Prints values from a sample spreadsheet.

  Initializes and returns Google Sheets API credentials.
  This function handles the authorization flow for accessing the Google Sheets API.
  It checks for existing credentials in the 'token.json' file. If valid credentials
  are found, they are loaded and returned. If the credentials are expired or not
  found, the user is prompted to log in, and new credentials are obtained and saved
  to 'token.json' for future use.
  Returns:
    Credentials | ExtCredentials | None: The authenticated credentials object, or
    None if the authentication process fails.
  Raises:
    FileNotFoundError: If the 'credentials.json' file is not found.
    google.auth.exceptions.GoogleAuthError: If there is an error during the
    authentication process.
  """
  creds = None
  # The file token.json stores the user's access and refresh tokens, and is
  # created automatically when the authorization flow completes for the first
  # time.
  if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
  # If there are no (valid) credentials available, let the user log in.
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      flow = InstalledAppFlow.from_client_secrets_file(
          "credentials.json", SCOPES
      )
      creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open("token.json", "w") as token:
      token.write(creds.to_json())
  return creds

class Spreadsheet():
  """
  A class to interact with Google Sheets API.

  To use this class, you need to have the `credentials.json` file, which contains
  your OAuth 2.0 client ID and client secret. You can obtain this file by creating
  a project in the Google Developers Console, enabling the Google Sheets API, and
  creating OAuth 2.0 credentials. Save the `credentials.json` file in the same
  directory as this script.

  When you run the script for the first time, it will prompt you to log in with
  your Google account and authorize the application to access your Google Sheets.
  After authorization, the script will save the access and refresh tokens in a
  file named `token.json`. This file will be used to authenticate subsequent runs
  of the script without requiring you to log in again.
  
  Methods:
    __init__(spreadsheet_id: str):
      Initializes the Spreadsheet instance with the given spreadsheet ID.
    get_rows(range_name: str) -> Sequence[dict[int, Any]]:
      Retrieves rows from the specified range in the spreadsheet.
  """

  def __init__(self, spreadsheet_id: str):
    creds = init_credentials()
    service = build("sheets", "v4", credentials=creds)

    # Call the Sheets API
    self._sheet = service.spreadsheets()
    self._spreadsheet_id = spreadsheet_id

  def get_rows(self, range_name: str) -> Sequence[dict[int, str]]:
    """
    Retrieves rows of data from a specified range in the spreadsheet.
    Args:
      range_name (str): The A1 notation of the range to retrieve values from.
    Returns:
      Sequence[dict[int, str]]: A sequence of dictionaries where each dictionary represents a row of data.
      If no data is found, an empty list is returned.
    Raises:
      googleapiclient.errors.HttpError: If an error occurs while fetching data from the Google Sheets API.
    """

    result = (
        self._sheet.values()
          .get(spreadsheetId=self._spreadsheet_id, range=range_name)
          .execute()
    )

    rows = result.get("values", [])

    if not rows:
      print("No data found.")
      return [] 
    
    return rows

  def write_rows(self, data: Sequence[dict[int, str]], range_name: str, input_option: InputValueOption):
    body = {
      'values': data
    }
    result = self._sheet.values().update(
      spreadsheetId=self._spreadsheet_id,
      range=range_name,
      valueInputOption=input_option,
      body=body
    ).execute()
    return result

def main():
  s = Spreadsheet(SAMPLE_SPREADSHEET_ID)
  values = s.get_rows(SAMPLE_RANGE_NAME)

  if not values:
    print("No data found.")
    return

  for row in values:
    # Print all of row B
    print(f"{row[1]}")

if __name__ == "__main__":
  logging.basicConfig(level=logging.DEBUG,
                      format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                      datefmt='%m-%d %H:%M')
  main()