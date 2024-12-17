SmartTrader README File

Team Details:

1. Name: Rahul Kanwal 
SJSU Email: rahul.kanwal@sjsu.edu
Student ID: 017626882

2. Name: Prem Jadhav
SJSU Email: premjitendra.jadhav@sjsu.edu
Student ID: 018172310

3. Name: Vandana Satish
SJSU Email: vandana.satish@sjsu.edu
Student ID: 018192941

4. Name: Kartik Chindarkar
SJSU Email: kartik.chindarkar@sjsu.edu
Student ID: 017583868
—-------------------------------------
Application URL:  https://smarttrader-ssta.onrender.com/
—-------------------------------------
Information to know:

Deployment on Free Tier: Our application is hosted on the free version of Render Cloud. As such, it may experience some delays in loading and processing requests.

Initial Load Time: Due to the nature of the free tier, the application instance may spin down during periods of inactivity. This can result in an initial delay of 50 seconds or more when you first access the application.

Processing Delays: After selecting a date and clicking the 'Predict' button to generate a stock trading strategy, please allow some additional time for the results to be processed and displayed. The duration may vary depending on current server load and activity levels.
—-------------------------------------
Build instructions:

Prerequisites: 
Python Installation: Ensure Python 3.x is installed on your system.

Setting Up the Environment:
Navigate to the Project Directory: Open a terminal and navigate to the “SmartTrader” directory where app.py is located.

Create and Activate a Virtual Environment -
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies -
pip install -r requirements.txt

Run the application:
Start the Flask development server by using the following command:
flask run

Accessing the Application:
Open a web browser and go to http://127.0.0.1:5000/ to access the application.
