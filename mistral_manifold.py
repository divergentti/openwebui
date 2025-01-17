"""
title: Mistral Manifold Pipe
author: Jari Hiltunen / Divergentti https://github.com/divergentti
modified from Anthropic function done by author_url: https://github.com/justinh-rahb
version: 0.0.1
required_open_webui_version: 0.3.17
license: MIT

Function lists all models at Mistral.ai and adds them to Models-selection.
If you are using free API, you can use mistral-nemo, mistral-Pixtral and Codestral Mamba.
Before using, create your free API-keys at https://console.mistral.ai/
After installing this Function, add your API-key to the Valves of the Function (gear rightmost)
"""

import os
import requests
import json
import time
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message

class Pipe:
    class Valves(BaseModel):
        MISTRAL_API_KEY: str = Field(default="")


    def __init__(self):
        self.type = "manifold"
        self.id = "mistral"
        self.name = "mistral/"
        # European server
        self.server= "https://api.mistral.ai"
        self.chat_url = self.server + "/v1/chat/completions"
        self.models_path = self.server + "/v1/models"

        self.temperature = 0.35
        self.top_p = 0.9
        self.max_tokens = 4096
        self.debug = True
        self.MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB per image

        # Read the environment variable and strip any whitespace
        api_key = os.getenv("MISTRAL_API_KEY", "").strip()
        self.valves = self.Valves(MISTRAL_API_KEY=api_key)
        if self.debug:
            print("API-key used:", self.valves.MISTRAL_API_KEY)  # Add this line for verification
        self.last_request_time = 0  # Initialize the last request time for rate limiting
        self.rate_limit_interval = 1  # Set the rate limit interval in seconds

    def get_mistral_models(self):
        headers = {
            "Authorization": f"Bearer {self.valves.MISTRAL_API_KEY}",
            "Content-Type": "application/json",
        }
        response = requests.get(f"{self.server}/v1/models", headers=headers)
        response.raise_for_status()
        models = response.json()["data"]
        if self.debug:
            print("Models:", models )
        return [{"id": model["id"], "name": model["name"]} for model in models]


    def pipes(self) -> List[dict]:
        models = self.get_mistral_models()
        return [{"id": model["id"], "name": f"mistral/{model['name']}"} for model in models]

    def process_image(self, image_data):
        """Process image data with size validation."""
        if image_data["image_url"]["url"].startswith("data:image"):
            mime_type, base64_data = image_data["image_url"]["url"].split(",", 1)
            media_type = mime_type.split(":")[1].split(";")[0]

            # Check base64 image size
            image_size = len(base64_data) * 3 / 4  # Convert base64 size to bytes
            if image_size > self.MAX_IMAGE_SIZE:
                raise ValueError(
                    f"Image size exceeds 5MB limit: {image_size / (1024 * 1024):.2f}MB"
                )

            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data,
                },
            }
        else:
            # For URL images, perform size check after fetching
            url = image_data["image_url"]["url"]
            response = requests.head(url, allow_redirects=True)
            content_length = int(response.headers.get("content-length", 0))

            if content_length > self.MAX_IMAGE_SIZE:
                raise ValueError(
                    f"Image at URL exceeds 5MB limit: {content_length / (1024 * 1024):.2f}MB"
                )

            return {
                "type": "image",
                "source": {"type": "url", "url": url},
            }


    def pipe(self, body: dict) -> Union[str, Generator, Iterator]:
        system_message, messages = pop_system_message(body["messages"])

        processed_messages = []
        total_image_size = 0

        for message in messages:
            processed_content = []
            if isinstance(message.get("content"), list):
                for item in message["content"]:
                    if item["type"] == "text":
                        processed_content.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "image_url":
                        processed_image = self.process_image(item)
                        processed_content.append(processed_image)

                        # Track total size for base64 images
                        if processed_image["source"]["type"] == "base64":
                            image_size = len(processed_image["source"]["data"]) * 3 / 4
                            total_image_size += image_size
                            if (
                                total_image_size > 100 * 1024 * 1024
                            ):  # 100MB total limit
                                raise ValueError(
                                    "Total size of images exceeds 100 MB limit"
                                )
            else:
                processed_content = [
                    {"type": "text", "text": message.get("content", "")}
                ]

            processed_messages.append(
                {"role": message["role"], "content": processed_content}
            )

        # payload is defined as it is at mistral.ai website
        payload = {
            "model": body["model"][body["model"].find(".") + 1 :],
            "temperature": body.get("temperature", self.temperature),
            "top_p": body.get("top_p", self.top_p),
            "max_tokens": body.get("max_tokens", self.max_tokens),
            "stream": body.get("stream", False),
            "messages": processed_messages,
        }

        headers = {
            "Authorization": f"Bearer {self.valves.MISTRAL_API_KEY}",
            "mistral-version": "2025-01-01",
            "Content-Type": "application/json",
        }

        if self.debug:
            print("Headers being sent   :", headers)  # Add this line for verification

        # Rate limiting
        current_time = time.time()
        elapsed_time = current_time - self.last_request_time
        if elapsed_time < self.rate_limit_interval:
            time.sleep(self.rate_limit_interval - elapsed_time)
        self.last_request_time = time.time()

        try:
            if body.get("stream", False):
                return self.stream_response(self.chat_url, headers, payload)
            else:
                return self.non_stream_response(self.chat_url, headers, payload)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return f"Error: Request failed: {e}"
        except Exception as e:
            print(f"Error in pipe method: {e}")
            return f"Error: {e}"

    def stream_response(self, url, headers, payload):
        headers["Authorization"] = f"Bearer {self.valves.MISTRAL_API_KEY}"
        try:
            with requests.post(
                url, headers=headers, json=payload, stream=True, timeout=(3.05, 60)
            ) as response:
                if response.status_code != 200:
                    raise Exception(
                        f"HTTP Error {response.status_code}: {response.text}"
                    )

                for line in response.iter_lines():
                    if line:
                        line = line.decode("utf-8")
                        if line == "data: [DONE]":
                            print("Streaming completed successfully.")
                            break
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                if data.get("choices"):
                                    for choice in data["choices"]:
                                        if "delta" in choice and "content" in choice["delta"]:
                                            yield choice["delta"]["content"]
                                        elif "finish_reason" in choice and choice["finish_reason"] == "stop":
                                            break

                                time.sleep(
                                    0.01
                                )  # Delay to avoid overwhelming the client

                            except json.JSONDecodeError:
                                print(f"Failed to parse JSON: {line}")
                            except KeyError as e:
                                print(f"Unexpected data structure: {e}")
                                print(f"Full data: {data}")
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            yield f"Error: Request failed: {e}"
        except Exception as e:
            print(f"General error in stream_response method: {e}")
            yield f"Error: {e}"

    def non_stream_response(self, url, headers, payload):
        headers["Authorization"] = f"Bearer {self.valves.MISTRAL_API_KEY}"
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=(3.05, 60)
            )
            if response.status_code != 200:
                raise Exception(f"HTTP Error {response.status_code}: {response.text}")

            res = response.json()
            return (
                res["choices"][0]["message"]["content"] if "choices" in res and res["choices"] else ""
            )
        except requests.exceptions.RequestException as e:
            print(f"Failed non-stream request: {e}")
            return f"Error: {e}"

