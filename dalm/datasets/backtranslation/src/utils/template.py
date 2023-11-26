from fastchat.conversation import Conversation, SeparatorStyle, register_conv_template

register_conv_template(
    Conversation(
        name="vicuna_v1.1_reverse",
        system_message=(
            "This is a chat between a curious user and a helpful artificial intelligence assistant. "
            "Given the assistant's reponse, please predict the user's instruction."
        ),
        roles=("RESPONSE", "INSTRUCTION"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

register_conv_template(
    Conversation(
        name="vicuna_v1.1_aug",
        system_message=(
            "This is a chat between a curious user and a helpful artificial intelligence assistant. "
            "Given the assistant's reponse, please predict the user's instruction in the style of an AI Assistant."
        ),
        roles=("USER", "ASSISTANT"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

register_conv_template(
    Conversation(
        name="vicuna_v1.1_seed",
        system_message=(
            "This is a chat between a curious user and a helpful artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions with knowledge from web search."
        ),
        roles=("USER", "ASSISTANT"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)
