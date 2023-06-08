import streamlit as st

st.markdown("""
    <style>
        body {
            background-image: url("https://img.freepik.com/free-photo/abstract-blur-green-nature_1150-27263.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .stApp {
            background-color: rgba(255, 255, 255, 0.1);
        }
    </style>
""", unsafe_allow_html=True)













st.markdown(
    """
    <style>
    .text {
        font-size: 90px;
        font-weight: bold;
        color: #1C249A;  /* Replace with your desired color code */
    }
    .team-name {
        font-size: 60px;
        font-weight: bold;
        color: #D0272C;  /* Replace with your desired color code */
    }
    .team-members-heading {
        font-size: 40px;
        margin-top: 40px;
        color: #141D98;  /* Replace with your desired color code */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the title and team name
st.markdown("<h1 class='text'>Plant Disease Classification</h1>", unsafe_allow_html=True)


team_name = "By : Neural Ninjas"
ninja_emoji = "🥷"  # Replace this with the Ninja emoji you prefer
markdown_text = f"<h2 class='team-name'>{team_name} {ninja_emoji}</h2>"
st.markdown(markdown_text, unsafe_allow_html=True)
