import io
import json
import pickle
import base64
import requests
import sys

class ModelStealer:
    def __init__(self, token: str, base_url: str = "http://34.122.51.94:9090"):
        self.token = token
        self.base_url = base_url
        self.port = None
        self.seed = None

    def request_new_api(self):
        try:
            response = requests.get(f"{self.base_url}/stealing_launch", headers={"token": self.token})
            answer = response.json()

            if 'detail' in answer:
                print(f"Error: {answer['detail']}")
                sys.exit(1)

            self.seed = str(answer['seed'])
            self.port = str(answer['port'])

            return self.seed, self.port

        except Exception as e:
            print(f"Exception occurred: {e}")
            sys.exit(1)

    def query_api(self, images, image_ids, idx, port):
        if port is None:
            raise ValueError("API port not set. Call `request_new_api()` first.")

        endpoint = "/query"
        url = f"http://34.122.51.94:{port}{endpoint}"
        image_data = []

        for img in images:
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            image_data.append(img_base64)

        payload = json.dumps(image_data)
        response = requests.get(url, files={"file": payload}, headers={"token": self.token})

        if response.status_code == 200:
            representations = response.json()["representations"]
            self._save_output(representations, image_ids, idx)
            return representations
        else:
            raise Exception(
                f"Model stealing failed. Code: {response.status_code}, content: {response.json()}"
            )
        
    def submit_model(self, seed, path):
        response = requests.post(self.base_url + "/stealing", files={"file": open(path, "rb")}, headers={"token": self.token, "seed": seed})
        print(response.json())

    def _save_output(self, representations, image_ids, idx):
        file_path = f"./results/out{idx}.pickle"
        with open(file_path, 'wb') as handle:
            pickle.dump({"ids": image_ids, "representations": representations}, handle, protocol=pickle.HIGHEST_PROTOCOL)
