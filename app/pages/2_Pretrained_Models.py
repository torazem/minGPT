import torch
from mingpt.bpe import BPETokenizer, get_encoder
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit.components.v1 as components

from mingpt.model import GPT, CausalSelfAttention

DEVICE = "cpu"
MODEL_TYPES = [None, "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]


def print_repr_func(expression, result):
    st.code(
        f"""
            >>> {expression}
            {result}
    """
    )


model_type = st.selectbox("Model type", MODEL_TYPES)
if model_type:
    model = GPT.from_pretrained(model_type)
    # ship model to device and set to eval mode
    model.to(DEVICE)
    model.eval()

    named_parameters = dict(name=[], shape=[], requires_grad=[])
    for name, parameter in model.named_parameters():
        named_parameters["name"].append(name.split("."))
        named_parameters["shape"].append(parameter.data.shape)
        named_parameters["requires_grad"].append(parameter.requires_grad)
    params_df = pd.DataFrame(named_parameters).set_index("name")


tab1, tab2, tab3 = st.tabs(
    ["Overview", "Configuration", "CausalSelfAttention Deep Dive"]
)

if model_type:
    with tab1:
        st.write(params_df)

    with tab2:
        pass
        # print_repr_func("model.n_head", model.n_head)
        # print_repr_func("model.n_embd", model.n_embd)

    with tab3:
        subtabs = st.tabs(["A"])
        with subtabs[0]:
            attn: CausalSelfAttention = model.get_submodule("transformer.h.0.attn")
            B = st.slider("Batch size", value=13, min_value=1, step=1)
            T = st.slider("Sequence length", value=7, min_value=1, step=1)
            C = 768

            input = torch.randn(B, T, C)
            output = attn(
                input,
                print_func=st.write,
                print_repr_func=print_repr_func,
                print_code_func=st.code,
            )
else:
    st.write("Select a model first")
