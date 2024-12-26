import requests
import zipfile
import io
from requests import Response


class CryptoDataset:
    url: str
    save_path = "./datasets"
    files_path: list[str] = []

    def __init__(self, url: str):
        self.url = url

    def download(self):
        response: Response = requests.get(self.url, stream = True)

        with zipfile.ZipFile(io.BytesIO(response.content)) as file:
            file.extractall(self.save_path)

            for name in file.namelist():
                self.files_path.append(f"{self.save_path}/{name}")
