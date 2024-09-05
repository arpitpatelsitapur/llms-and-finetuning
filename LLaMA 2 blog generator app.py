import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

## response function
def get_llama2_response(input_text,words_count,blog_style):
    
    ## calling llama2 model
    llm=CTransformers(
        model='models/llama-2-7b-chat.ggmlv3.q4_0.bin',
        model_type='llama',
        config={
            'max_new_tokens':256,
            'temperature':0.01
            }
        )
    
    ## Prompt template
    template="""
        Write a suitable and usable Blog for {blog_style} for the topic {input_text} within {words_count} words.
        """
    prompt=PromptTemplate(input_variables=['style','text','n_words'],
                          template=template)
    
    ## generate response using llama 2
    response=llm(prompt.format(
        blog_style=blog_style,
        input_text=input_text,
        words_count=words_count))
    print(response)
    return response
    




## app ui
st.set_page_config(
    page_icon='🤖',
    layout='centered',
    initial_sidebar_state='collapsed'
)

st.header("Generate Blogs using AI 🤖")

input_text=st.text_area('Enter Blog Topic.')


col1,col2=st.columns([5,5])
with col1:
    words_count=st.text_input('Words count')
with col2:
    blog_style=st.selectbox(
        'Write Blog for',
        ('Common user','Researchers','Developers'),
        index=0)    


submit=st.button('Generate')

## final reponse
if submit:
    st.write(get_llama2_response(input_text,words_count,blog_style))



