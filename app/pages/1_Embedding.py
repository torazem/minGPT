from mingpt.bpe import get_encoder
import streamlit as st

def _bytes_to_hex_str(token_bytes):
    """
    Convert bytes to hex in the format I like.

    For example b' \xf0\x9f\xa4\x97,' -> "20 F0 9F A4 97 2C"
    """
    hex_bytes = [hex(b).replace("0x", "").upper() for b in token_bytes]
    return " ".join(hex_bytes)


# I added "discombobulated" to Karpathy's example to see how complex words
# are broken into prefixes and suffixes and other little bits.
prompt = st.text_input("Prompt:", value="Hello!!!!!!!!!? 🤗, my dog is a little discombobulated")

# Setup
enc = get_encoder()
encode_work = enc.encode_and_show_work(prompt)

tab1, tab2 = st.tabs(["Pre-tokenization", "Byte-pair encoding"])

with tab1:
    st.write("Pre-tokenization is done with simple regex", encode_work["tokens"])

with tab2:
    st.write("""
    ### Byte-pair encoding (BPE)
    Byte-pair encoding (BPE) is a tokenization technique that uses a pre-populated dictionary (vocabulary) of tokens to split a sentence into recognised parts.
    The vocabulary is built on sequences of characters (bytes) that are ranked from most to least common.

    > **Side note**: Interestingly, minGPT converts each raw byte into a readable character before BPE. This isn't strictly necessary, but is very useful for debugging.

    Below is a breakdown of how BPE works using the pre-tokens extracted from the prompt.
    """)
    st.write("#### Interactive BPE")
    selected_token = st.select_slider(
        "Pick a pre-token",
        list(range(len(encode_work["tokens"]))),
        format_func=lambda x: encode_work["tokens"][x]
    )
    parts = encode_work["parts"][selected_token]
    _, bpe_work = enc.bpe_and_show_work(parts["token_translated"])
    st.write(
        {
            "bytes": parts['token_bytes'],
            "bytes (hex)": _bytes_to_hex_str(parts["token_bytes"]),
            "transcribed": parts['token_translated']
        }
    )
    st.write("""
    BPE splits the pre-token into pairs, starting with pairs of individual characters/bytes.
    The pairs are merged with adjacent letters, and the BPE vocabulary is checked to see if this merge makes a valid token.
    If the pair doesn't exist in the BPE vocabulary, the individual bytes are used as tokens.
    """)

    st.write("For this pre-token, BPE found the following tokens in its vocabulary:", parts["token_merged"])
    st.write("...which are then converted to indexes:", parts["token_ix"])

    st.write(f"BPE for this pre-token was completed in **{len(bpe_work)}** steps.")
    with st.expander("See details"):
        st.write("At each step, the most common 'bigram' (pair of token candidates) is selected for processing")
        for i, step in enumerate(bpe_work):
            st.write(f"#### Step {i}")
            st.write("Word:", step["word"])
            st.write("Pairs:", step["pairs"])
            st.write("Selected bigram:", step["bigram"])
