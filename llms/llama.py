import llamaapi

class Llama:
    def __init__(self, api_key, version):
        self.api_key = api_key
        self.version = version

    def submit(self, prompt, temperature=1, num_responses=3) -> list:
        """
        Submit a prompt to Llama.
        :param prompt: Prompt for Llama.
        :return: List of length `num_responses` containing the LLM's output in response to the prompt.
        """
        llama = llamaapi.LlamaAPI(self.api_key)

        # Build the API request
        api_request_json = {
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "model": self.version
        }

        responses = [llama.run(api_request_json).json()['choices'][0]['message']['content'] for _ in range(num_responses)]

        return responses
