from pytube import YouTube
from haystack.nodes import PromptNode, PromptModel
from haystack.nodes.audio import WhisperTranscriber
from haystack.pipelines import Pipeline
from model_add import LlamaCPPInvocationLayer

summary_prompt = "deepset/summarization"

def youtube2audio (url: str):
    yt = YouTube(url)
    video = yt.streams.filter(abr='160kbps').last()
    return video.download()  

whisper = WhisperTranscriber()

full_path = "llama-2-7b-32k-instruct.Q4_K_S.gguf"

model = PromptModel(model_name_or_path=full_path, invocation_layer_class=LlamaCPPInvocationLayer, use_gpu=False, max_length=512)

print(model)

prompt_node = PromptNode(model_name_or_path=model, default_prompt_template=summary_prompt, use_gpu=False)

print("###############################################")

print(prompt_node)

print("###############################################")

file_path = youtube2audio("https://www.youtube.com/watch?v=h5id4erwD4s")

pipeline = Pipeline()
pipeline.add_node(component=whisper, name="whisper", inputs=["File"])
pipeline.add_node(component=prompt_node, name="prompt", inputs=["whisper"])

output = pipeline.run(file_paths=[file_path])

print(output["results"])

print(output["results"][0].split("\n\n[INST]")[0])
