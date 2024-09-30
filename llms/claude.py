import anthropic

class Claude:
    def __init__(self, api_key, version="claude-3-5-sonnet-20240620"):
        self.api_key = api_key
        self.version = version

    def submit(self, prompt, temperature=1, num_responses=3, max_tokens=1024) -> list:
        """
        Submit a prompt to Claude.
        :param prompt: Prompt for Claude.
        :return: List of length `num_responses` containing the LLM's output in response to the prompt.
        """
        client = anthropic.Client(api_key=self.api_key)

        responses = []
        for i in range(num_responses):
            message = client.messages.create(
                model=self.version,
                max_tokens=max_tokens,
                temperature=temperature,
                # num_completions = 3,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }]}])

            # Save response
            responses.append(message.content[0].text)

        return responses
