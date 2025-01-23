# title: Mistral Agents Pipe
# authors: Jari Hiltunen
# author_url: https://github.com/divergentti
# version: 0.0.1 (23.01.2025)
# required_open_webui_version: 0.3.17
# license: MIT
#
# Before using, create your free API-keys at https://console.mistral.ai/api-keys/
# Create, educate and fine tune your agent here https://console.mistral.ai/build/agents
# Pick the agent ID from the page, begins with ag:
# After installing this Function, add your API-key to the Valves of the Function (gear rightmost)
# Use @ to pick up the agent to your conversation in Open WebUI
import os
import requests
import json
import time
from typing import List, Union, Generator, Iterator, Dict
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message


class Pipe:
    class Valves(BaseModel):
        MISTRAL_API_KEY: str = Field(default="")

    def __init__(self):
        self.debug_stream = True
        self.debug_tokens = True
        self.debug_errors = True
        self.type = "manifold"
        self.id = "mistral-agents"
        self.name = "agent/"
        self.agents_endpoint = "https://api.mistral.ai/v1/agents/completions"
        self.agent_1_id = "ag:xxxx:social-work-and-empathy-agent:xxx8a8"  # Pick this from Mistral
        self.agent_1_name = "Social Work"  # This is name listed in agents and callable with @-sign during chat
        self.top_p = 0.9
        self.max_tokens = 4096
        self.max_tokens_minute = 500000  # Tokens per minute
        self.tokens_left = (
            1000000000  # Will be updated from stream, this is max per month
        )
        self.total_tokens_used = 0
        api_key = os.getenv("MISTRAL_API_KEY", "").strip()
        self.valves = self.Valves(MISTRAL_API_KEY=api_key)
        self.last_request_time: float = (
            0.0  # Initialize the last request time for rate limiting
        )
        self.rate_limit_reset: float = 0.0  # Initialize rate_limit_reset to 0
        self.rate_limit_interval: float = (
            30.0  # Set the rate limit interval in seconds (Open is 100 request per hour)
        )
        self.last_token_reset_time = time.time()  # Initialize the last token reset time
        self.tokens_used_this_minute = 0  # Initialize tokens used this minute

    def pipes(self) -> List[dict]:
        if self.agent_1_id:
            return [{"id": self.agent_1_id, "name": self.agent_1_name}]

    def pipe(self, body: dict) -> Union[str, Generator, Iterator]:
        system_message, messages = pop_system_message(body["messages"])
        processed_messages = []
        for message in messages:
            processed_content = []
            if isinstance(message.get("content"), list):
                for item in message["content"]:
                    if item["type"] == "text":
                        processed_content.append({"type": "text", "text": item["text"]})
            else:
                processed_content = [
                    {"type": "text", "text": message.get("content", "")}
                ]
            processed_messages.append(
                {"role": message["role"], "content": processed_content}
            )
        # payload is defined as it is at mistral.ai website
        payload = {
            "agent_id": self.agent_1_id,
            "top_p": body.get("top_p", self.top_p),
            "max_tokens": body.get("max_tokens", self.max_tokens),
            "stream": body.get("stream", False),
            "messages": processed_messages,
        }
        headers = {
            "Authorization": f"Bearer {self.valves.MISTRAL_API_KEY}",
            "mistral-version": "2025-01-01",
            "Content-Type": "application/json",
            "user-agent": "MistralAgent/1.0",
            "accept": "application/json",
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
                return self.stream_response(self.agents_endpoint, headers, payload)
            else:
                return self.non_stream_response(self.agents_endpoint, headers, payload)
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
                for line in response.iter_lines():Code you need to modify
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
                                        if (
                                            "delta" in choice
                                            and "content" in choice["delta"]
                                        ):
                                            yield choice["delta"]["content"]
                                        elif (
                                            "finish_reason" in choice
                                            and choice["finish_reason"] == "stop"
                                        ):
                                            break
                                    # Extract token usage information
                                    usage = data.get("usage", {})
                                    prompt_tokens = usage.get("prompt_tokens", 0)
                                    completion_tokens = usage.get(
                                        "completion_tokens", 0
                                    )
                                    total_tokens = usage.get("total_tokens", 0)
                                    # Update total tokens used
                                    self.total_tokens_used += total_tokens
                                    self.tokens_left -= total_tokens
                                    # Update tokens used this minute
                                    self.tokens_used_this_minute += total_tokens
                                    # Check if a minute has passed
                                    if time.time() - self.last_token_reset_time >= 60:
                                        self.tokens_used_this_minute = 0
                                        self.last_token_reset_time = time.time()
                                    if self.debug_tokens and total_tokens > 0:
                                        print(f"Prompt Tokens: {prompt_tokens}")
                                        print(f"Completion Tokens: {completion_tokens}")
                                        print(f"Total Tokens: {total_tokens}")
                                        print(
                                            f"Total Tokens Used: {self.total_tokens_used}"
                                        )
                                        print(f"Tokens Left: {self.tokens_left}")
                                        print(
                                            f"Tokens Used This Minute: {self.tokens_used_this_minute}"
                                        )
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
                res["choices"][0]["message"]["content"]
                if "choices" in res and res["choices"]
                else ""
            )
        except requests.exceptions.RequestException as e:
            if self.debug_stream:
                print(f"Failed non-stream request: {e}")
            return f"Error: {e}"
