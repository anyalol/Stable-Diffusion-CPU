import gradio as gr



def process(name) :
  return "hello " + name
  
gr.Interface(fn=process, inputs="text", outputs="text").launch()
