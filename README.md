# Environment and Installation

This project is tested under **Python 3.10** and **PyTorch 2.2.1**.

**1. Prepare Project Files**
* Download all project files and folders, and unzip them to your local working directory.
* In PyCharm, select `File` > `Open...`, and then choose the project folder you just unzipped.

**2. Configure Python Interpreter**
* In PyCharm, go to `File` > `Settings` > `Project: [Your Project Name]` > `Python Interpreter`.
* From the interpreter list, select your locally installed Python interpreter (e.g., Python 3.10).

**3. Install Dependencies**
* Open the **Terminal** window at the bottom of PyCharm.
* Run the following command to install all project dependencies:
    ```bash
    pip install -r requirements.txt
    ```

# Running the Test

This project includes all necessary datasets and the pre-trained model. After completing the environment setup and dependency installation, you can directly run the evaluation script.

* In the PyCharm **Terminal**, execute the following command:
    ```bash
    python evaluate.py
    ```
* After the script runs, it will output the model's performance metrics on the test set to the console.
