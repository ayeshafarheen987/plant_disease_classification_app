import streamlit as st
st.markdown("""
    <style>
        body {
            background-image: url("https://e0.pxfuel.com/wallpapers/200/934/desktop-wallpaper-sparkly-background-30-background-sparkle.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .stApp {
            background-color: rgba(255, 255, 255, 0.1);
        }
    </style>
""", unsafe_allow_html=True)








# Define team member information
team_members = [
    {
        "name": "Kanav Bansal",
        "role": "Chief Data Scientist",
        "linkedin": "https://www.linkedin.com/in/kanavbansal/"
    },
    {
        "name": "Meher Fatima",
        "role": "Project Manager",
        "linkedin": "https://www.linkedin.com/in/meher-fatima-aa3b3a37/"
    },
    {
        "name": "Abdul Jaweed",
        "role": "Team Lead",
        "linkedin": "https://www.linkedin.com/in/abdul-jaweed-datascientist/"
    },
    {
        "name": "Ayesha Farheen",
        "role": "Team Member",
        "linkedin": "https://www.linkedin.com/in/ayeshafarheen-/"
    },
    {
        "name": "Masna Ashraf",
        "role": "Team Member",
        "linkedin": "https://www.linkedin.com/in/masna-ashraf/"
    },
    {
        "name": "Chandni Kumari",
        "role": "Team Member",
        "linkedin": "https://www.linkedin.com/in/chandniikumari/"
    },
    {
        "name": "Mohneet Kaur",
        "role": "Team Member",
        "linkedin": "https://www.linkedin.com/in/mohneet-kaur"
    },
    {
        "name": "Yasasvini",
        "role": "Team Member",
        "linkedin": "https://www.linkedin.com/in/yasasvini"
    }
]

# Page title
st.title(":rainbow: Team Members")

# Display team member information
for member in team_members:
    st.subheader(f":star: {member['name']}")
    st.write(f"Role: {member['role']}")
    st.markdown(f"LinkedIn: [{member['name']}]({member['linkedin']})")
    st.markdown("---")

# Thank you message
st.title(":sparkles: Thank you for taking a look at our application and experimenting!")
 