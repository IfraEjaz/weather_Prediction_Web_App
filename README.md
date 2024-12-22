
Here’s the copyable version of the README with the commands clearly marked in bold:

markdown
Copy code
# Weather Prediction Web App

This is a Django-based web application for weather prediction. Below are the steps to set up and run the project on your local machine.

## Prerequisites

Before starting the project, make sure you have the following installed:

- **Python 3.x**: [Download Python](https://www.python.org/downloads/)
- **pip**: Python package installer (should be installed with Python)
- **Virtualenv** (optional but recommended): `pip install virtualenv`

## Step 1: Clone the Repository

First, clone the repository to your local machine:

**Clone the repository:**
```bash
git clone git@github.com:IfraEjaz/weather_Prediction_Web_App.git
cd weather_Prediction_Web_App
Step 2: Install Dependencies
It is recommended to set up a virtual environment for the project to avoid conflicts with other Python projects on your system. If you're not using a virtual environment, you can skip this step.

Create a Virtual Environment (Optional but Recommended)
Create a virtual environment:

bash
Copy code
python -m venv venv
Activate the virtual environment:

Windows:
bash
Copy code
venv\Scripts\activate
Install Required Packages
Next, install the required packages listed in the requirements.txt file:

bash
Copy code
pip install -r requirements.txt
Step 3: Set Up the Database
Before running the project, set up the database by running the Django migrations:

bash
Copy code
python manage.py migrate
Step 4: Run the Development Server
Finally, start the Django development server:

bash
Copy code
python manage.py runserver
You should now be able to access the web application by navigating to http://127.0.0.1:8000/ in your browser.

Troubleshooting
If you get an error about missing packages or modules, run the following command to install them:

bash
Copy code
pip install -r requirements.txt
If the project is not starting, ensure that all the necessary environment variables (like Django settings) are properly configured.

Optional: Install Django on Windows
If you need to install Django on Windows, follow these steps:

Install Django using pip:

bash
Copy code
pip install django
Verify the Installation: After installing Django, verify that it was successfully installed by running:

bash
Copy code
django-admin --version
Feel free to edit or expand the instructions according to your specific project setup. Let me know if you need more adjustments!

yaml
Copy code

---

This version is ready to be copied directly into your `README.md`. The command
