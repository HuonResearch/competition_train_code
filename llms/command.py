import cohere

class Command:
    def __init__(self, api_key, version):
        self.api_key = api_key
        self.version = version

    def submit(self, prompt, temperature=1, num_responses=3) -> list:
        """
        Submit a prompt to Command.
        :param prompt: Prompt for Command.
        :return: List of length `num_responses` containing the LLM's output in response to the prompt.
        """
        co = cohere.Client(api_key=self.api_key)

        responses = [co.chat(message=prompt, model=self.version, temperature=temperature).text for _ in range(num_responses)]
        return responses
