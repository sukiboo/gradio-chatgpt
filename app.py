import os
import time
import gradio as gr
from openai import OpenAI


class GPT:
    """Instantiate GPT model for a multi-step conversation."""

    def __init__(self, **params):
        """Setup model parameters and system prompt."""
        self.params = {'model': 'gpt-3.5-turbo', **params}
        self.messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]

    def chat(self, user_prompt, history=None, *params):
        """Update model parameters and generate a response."""
        # parse and update the parameters
        _, system_prompt, temperature, top_p, frequency_penalty, presence_penalty = params
        self.params['temperature'] = temperature
        self.params['top_p'] = top_p
        self.params['frequency_penalty'] = frequency_penalty
        self.params['presence_penalty'] = presence_penalty

        # update the message buffer and get a response
        self.messages[0]['content'] = f'{system_prompt}'
        self.messages.append({'role': 'user', 'content': f'{user_prompt}'})
        completion = client.chat.completions.create(messages=self.messages, **self.params)
        response = completion.choices[0].message.content
        self.messages.append({'role': 'assistant', 'content': response})

        return response


def create_chatbot(gpt):
    """Configure chatbot interface."""
    with gr.Blocks() as chatbot:

        # parameters accordion
        info = gr.Markdown('For additional parameters see [OpenAI Chat API Reference](https://platform.openai.com/docs/api-reference/chat)',
                           render=False)
        system_prompt = gr.Textbox('You are a helpful assistant.', label='system prompt', render=False)
        temperature = gr.Slider(0., 2., value=1., step=.1, label='temperature', render=False)
        top_p = gr.Slider(0., 1., value=1., step=.01, label='top_p', render=False)
        frequency_penalty = gr.Slider(-2., 2., value=0, step=.1, label='frequency_penalty', render=False)
        presence_penalty = gr.Slider(-2., 2., value=0, step=.1, label='presence_penalty', render=False)

        # chatbot interface
        gr.ChatInterface(
            fn=gpt.chat,
            title='ChatGPT',
            description='A simple implementation of ChatGPT',
            chatbot=gr.Chatbot(height=600, layout='bubble', render=False),
            textbox=gr.Textbox(placeholder='Message ChatGPT...', scale=9, render=False),
            additional_inputs_accordion=gr.Accordion(label='Parameters', open=False, render=False),
            additional_inputs=[info, system_prompt,
                               temperature, top_p,
                               frequency_penalty, presence_penalty],
            retry_btn=None,
            undo_btn=None,
            clear_btn=None,
            concurrency_limit=None,
            theme=None,
        )

    return chatbot


if __name__ == '__main__':

    client = OpenAI()
    gr.close_all()
    gpt = GPT()
    chatbot = create_chatbot(gpt)
    chatbot.launch()

