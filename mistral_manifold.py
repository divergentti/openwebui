"""
Draft Function for OpenWebUi

title: Mistral Manifold Pipe
author: Jari Hiltunen / Divergentti https://github.com/divergentti
modified from Anthropic function done by author_url: https://github.com/justinh-rahb
version: 0.0.4 (20.01.2025) - not continuing to develop further
required_open_webui_version: 0.3.17
license: MIT

Function lists all models at Mistral.ai and adds them to Models-selection and then tries to filter those
which are multiple times with different versions. This seems to be too complex approach. Define your own models
to the list as you like.

If you are using free API, you can use mistral-nemo, mistral-Pixtral and Codestral Mamba.
The free tier at Mistral AI allows for 100 requests per hour, which translates to approximately 1.67 requests
per minute or around 0.027 requests per second. To stay within the rate limit, you should adjust your rate
limit interval accordingly.

Before using, create your free API-keys at https://console.mistral.ai/
After installing this Function, add your API-key to the Valves of the Function (gear rightmost)

Note! Image creation is not tested
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
        self.debug_models = False
        self.debug_stream = True
        self.debug_errors = True
        self.type = "manifold"
        self.id = "mistral"
        self.name = "mistral/"
        # European server
        self.server= "https://api.mistral.ai"
        self.models_url = self.server + "/v1/models"
        self.chat_url = self.server + "/v1/chat/completions"
        self.temperature = 0.7
        self.top_p = 0.9
        self.max_tokens = 4096
        api_key = os.getenv("MISTRAL_API_KEY", "").strip()
        self.valves = self.Valves(MISTRAL_API_KEY=api_key)
        self.last_request_time: float = 0.0  # Initialize the last request time for rate limiting
        self.rate_limit_reset: float = 0.0  # Initialize rate_limit_reset to 0
        self.rate_limit_interval: float = 30.0  # Set the rate limit interval in seconds (Open is 100 request per hour)
        self.models = ""

        # Not yet implemented!
        self.MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB per image
        self.image_url = ""

    def get_mistral_models(self):
        if not self.valves.MISTRAL_API_KEY:
            raise ValueError("MISTRAL_API_KEY is missing or invalid.")
        headers = {
            "Authorization": f"Bearer {self.valves.MISTRAL_API_KEY}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.get(f"{self.models_url}", headers=headers)
            response.raise_for_status()
            self.models = response.json()["data"]
        except requests.exceptions.RequestException as e:
            if self.debug_errors:
                print(f"API call failed: {e}")

        # Map to track unique models
        model_map = {}
        for model in self.models:
            # Check if the model has the `completion_chat` capability
            if not model["capabilities"].get("completion_chat", False):
                continue

            # Extract base ID and check if it's a "latest" version
            base_id = "-".join(model["id"].split("-")[:-1])
            is_latest = "latest" in model["id"] or "latest" in model["aliases"]

            # Update or add model to the map
            if base_id not in model_map or is_latest:
                model_map[base_id] = model

        # Prepare the final list of unique models
        unique_models = []
        for base_id, model in model_map.items():
            unique_models.append({
                "id": model["id"],
                "name": model["name"],
                "capabilities": model["capabilities"],
                "description": model["description"],
                "max_context_length": model["max_context_length"],
                "aliases": model["aliases"],
                "deprecation": model["deprecation"],
                "default_model_temperature": model["default_model_temperature"],
                "type": model["type"],
            })

        if self.debug_models:
            print("Unique Models:")
            for model in unique_models:
                print(f"ID: {model['id']}")
                print(f"Name: {model['name']}")
                print(f"Capabilities: {model['capabilities']}")
                print(f"Description: {model['description']}")
                print(f"Max Context Length: {model['max_context_length']}")
                print(f"Aliases: {model['aliases']}")
                print(f"Deprecation: {model['deprecation']}")
                print(f"Default Model Temperature: {model['default_model_temperature']}")
                print(f"Type: {model['type']}")
                print("-" * 40)

        return unique_models

    def pipes(self) -> List[dict]:
        # This initiates the sub, but for some reason object (self) values are not passed further
        try:
            models = self.get_mistral_models()
            return [{"id": model["id"], "name": model["name"]} for model in models]
        except Exception as e:
            if self.debug_errors:
                print(f"Error fetching models: {e}")

    def process_image(self, image_data):
        if image_data["type"] == "image_url":
            url = image_data["url"]
            response = requests.head(url, allow_redirects=True)
            content_length = int(response.headers.get("content-length", 0))
            if content_length > self.MAX_IMAGE_SIZE:
                raise ValueError(f"Image at URL exceeds {self.MAX_IMAGE_SIZE / (1024 * 1024):.2f}MB limit")
            return {
                "type": "image_url",
                "url": url,
            }
        elif image_data["type"] == "image_base64":
            mime_type, base64_data = image_data["data"].split(",", 1)
            media_type = mime_type.split(":")[1].split(";")[0]
            image_size = len(base64_data) * 3 / 4  # Convert base64 size to bytes
            if image_size > self.MAX_IMAGE_SIZE:
                raise ValueError(f"Image size exceeds {self.MAX_IMAGE_SIZE / (1024 * 1024):.2f}MB limit")
            return {
                "type": "image_base64",
                "media_type": media_type,
                "data": base64_data,
            }
        elif image_data["type"] == "image_generate":
            prompt = image_data["prompt"]
            headers = {
                "Authorization": f"Bearer {self.valves.MISTRAL_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "prompt": prompt,
                "n_iter": 1,
                "size": "256x256",
            }
            try:
                response = requests.post(self.image_url, headers=headers, json=payload, timeout=(3.05, 60))
                response.raise_for_status()
                # Check rate limit headers
                rate_limit_remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
                rate_limit_reset = int(response.headers.get("X-RateLimit-Reset", 0))
                self.rate_limit_reset = rate_limit_reset
                if rate_limit_remaining == 0:
                    sleep_time: float = max(0.0, float(self.rate_limit_reset) - time.time())
                    if self.debug_stream:
                        print(f"Rate limit exceeded. Sleeping for {sleep_time:.2f} seconds.")
                    time.sleep(sleep_time)  # Note! This is not async method!

                image_data = response.json()["data"][0]
                image_url = image_data["url"]
                return {
                    "type": "image_generate",
                    "url": image_url,
                }
            except requests.exceptions.RequestException as e:
                if self.debug_stream:
                    print(f"Image generation failed: {e}")
                raise ValueError("Image generation failed")
            except requests.exceptions.RequestException as e:
                if self.debug_stream:
                    print(f"Rate limit header error: {e}")

        else:
            raise ValueError("Unsupported image type")


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

        if self.debug_stream:
            print("Headers being sent   :", headers)

        # Rate limiting
        current_time = time.time()
        elapsed_time: float = current_time - self.last_request_time
        if elapsed_time < self.rate_limit_interval:
            sleep_time: float = max(0.0, float(self.rate_limit_reset) - time.time())
            if self.debug_stream:
                print(f"Rate limit exceeded. Sleeping for {sleep_time:.2f} seconds.")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

        try:
            if body.get("stream", False):
                return self.stream_response(self.chat_url, headers, payload)
            else:
                return self.non_stream_response(self.chat_url, headers, payload)
        except requests.exceptions.RequestException as e:
            if self.debug_stream:
                print(f"Request failed: {e}")
            return f"Error: Request failed: {e}"
        except Exception as e:
            if self.debug_stream:
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
                    if self.debug_stream:
                        print("Response line: ", line)
                    if line:
                        line = line.decode("utf-8")
                        if line == "data: [DONE]":
                            if self.debug_stream:
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
                                if self.debug_stream:
                                    print(f"Failed to parse JSON: {line}")
                            except KeyError as e:
                                if self.debug_stream:
                                    print(f"Unexpected data structure: {e}")
                                    print(f"Full data: {data}")
        except requests.exceptions.RequestException as e:
            if self.debug_stream:
                print(f"Request failed: {e}")
            yield f"Error: Request failed: {e}"
        except Exception as e:
            if self.debug_stream:
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
            if self.debug_stream:
                print(f"Failed non-stream request: {e}")
            return f"Error: {e}"
