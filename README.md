## Weather Prediction Web App

This is a Django-based web application for weather prediction. Below are the steps to set up and run the project on your local machine.

### Prerequisites

Before starting the project, make sure you have the following installed:

- **Python 3.x**: [Download Python](https://www.python.org/downloads/)
- **pip**: Python package installer (should be installed with Python)
- **Virtualenv (optional but recommended)**: 
  ```bash
   pip install virtualenv
 
## Step 1: Clone the Repository
First, clone the repository to your local machine:
  ```bash
git clone git@github.com:IfraEjaz/weather_Prediction_Web_App.git
```

```bash
cd weather_Prediction_Web_App
```
## Step 2: Install Dependencies
It is recommended to set up a virtual environment for the project to avoid conflicts with other Python projects on your system. If you're not using a virtual environment, you can skip this step.

Create a Virtual Environment (Optional but Recommended)

Create a virtual environment:
  ```bash
python -m venv venv
```

Activate the virtual environment:
  ```bash
venv\Scripts\activate
```
Install Required Packages
Next, install the required packages listed in the requirements.txt file:
  ```bash
pip install -r requirements.txt
```
or 
  ```bash
python install -r requirements.txt
```
## Step 3: Set Up the Database

Before running the project, set up the database by running the Django migrations:
  ```bash
python manage.py migrate
```
## Step 4: Run the Development Server
Finally, start the Django development server:
  ```bash
python manage.py runserver
```
You should now be able to access the web application by navigating to **http://127.0.0.1:8000/** in your browser.




