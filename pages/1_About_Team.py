import streamlit as st
st.markdown("""
    <style>
        body {
            background-image: url("https://images.pexels.com/photos/255379/pexels-photo-255379.jpeg?cs=srgb&dl=pexels-miguel-%C3%A1-padri%C3%B1%C3%A1n-255379.jpg&fm=jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .stApp {
            background-color: rgba(255, 255, 255, 0.3);
        }
    </style>
""", unsafe_allow_html=True)








def main():
    # Define team members and their information
    team_members = [
         {
            "name": "Abdul Jaweed",
            "role": "Data Scientist",
            "responsibility": "MLOps & AWS Deployment",
            "bio": "Abdul Jaweed is a skilled Data Scientist with expertise in MLOps and AWS deployment. He specializes in designing and implementing efficient machine learning pipelines and deploying models on AWS infrastructure. Abdul's passion for leveraging advanced technologies and his proficiency in MLOps ensure streamlined model development, deployment, and monitoring. With his expertise, Abdul is dedicated to optimizing data-driven workflows and driving impactful business outcomes."

        },




        {
            "name": "Ayesha Farheen",
            "role": "Data Scientist",
            "responsibility": "Exploratory Data Analysis and deployment on Streamlit",
            "bio": "Ayesha is a talented Data Scientist with expertise in Python and server management. She excels in performing in-depth Exploratory Data Analysis to uncover insights and patterns.  She has a strong passion for transforming raw data into actionable insights and driving data-powered decision making."
        },

        {
            "name": "Masna Ashraf",
            "role": "Data Scientist",
            "responsibility": "Chatbot implementation",
            "bio": "Masna Ashraf is a talented Data Scientist with a passion for chatbot implementation. With her expertise in modern web technologies, Masna excels in developing and deploying interactive chatbot solutions.Masna's strong problem-solving skills and innovative mindset make her adept at designing chatbot systems that provide seamless and personalized user experiences."
        },


        {
            "name": "Chandni Kumari",
            "role": "Data Scientist",
            "responsibility": "UI/UX designing of web app",
            "bio": "Chandni Kumari is a passionate Data Scientist with a specialization in UI/UX designing of web apps. With a deep understanding of user-centered design principles, Chandni is dedicated to creating seamless and enjoyable user experiences.  With her passion for blending data-driven insights with engaging design, Chandni plays a crucial role in delivering exceptional user experiences in web applications."
        },

        {
            "name": "Mohneet Kaur",
            "role": "   ",
            "responsibility": "  ",
            "bio": "  ",
        },
        {
            "name": "Yasavini",
            "role": "  ",
            "responsibility": "  ",
            "bio": "  ",
        }
    ]

    # Set CSS styles
    css = """
    <style>
        .team-name {
            font-weight: bold;
            font-size: 48px;
            color: #210F76;
            text-align: center;
            margin-bottom: 20px;
        }
        .team-member {
            margin-bottom: 30px;
            padding: 20px;
            border: 1px solid #C6C6C6;
            border-radius: 10px;
        }
        .team-member-name {
            font-weight: bold;
            font-size: 24px;
            color: #210F76;
            margin-bottom: 10px;
        }
        .team-member-role {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .team-member-responsibility {
            margin-bottom: 5px;
        }
        .team-member-bio {
            margin-bottom: 10px;
        }
    </style>
    """

    # Render page
    st.markdown(css, unsafe_allow_html=True)
    st.markdown("<h1 class='team-name'>Neural Ninjas</h1>", unsafe_allow_html=True)

    for member in team_members:
        if member["name"] and member["role"] and member["responsibility"] and member["bio"]:
            st.markdown("<div class='team-member'>", unsafe_allow_html=True)
            st.markdown("<h2 class='team-member-name'>{}</h2>".format(member["name"]), unsafe_allow_html=True)
            st.markdown("<p class='team-member-role'>{}</p>".format(member["role"]), unsafe_allow_html=True)
            st.markdown("<p class='team-member-responsibility'>{}</p>".format(member["responsibility"]), unsafe_allow_html=True)
            st.markdown("<p class='team-member-bio'>{}</p>".format(member["bio"]), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
