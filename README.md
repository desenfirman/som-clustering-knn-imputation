# SOM Clustering with KNN Data Imputation

Source code for my undergraduate thesis — Region Grouping in East Java based on Person with Social Welfare Problems using Self-Organizing Maps Algorithm and K-Nearest Neighbors Missing-Value Imputation.

## Prerequisites

- Git
- A *.csv file to be clustered
- Conda (This project using Conda as an environment)
- A cup of coffee ☕

## Set-up app into your machine

1. Clone this repository into your machine

    ```bash
    git clone https://github.com/desenfirman/som-clustering-knn-imputation.git
    cd som-clustering-knn-imputation
    ```

2. Set up conda environment for this project.

    ```bash
    conda env create -f environment.yml
    ```

3. Wait for download and installation package completed. Drink-your-coffee. . . ☕
4. After installation completed, run this command to start a Flask webserver.

    ```bash
    python runwebserver.py
    ```

5. Access localhost:8000 to your browser and you're ready to use this app.

## How to use

1. Open localhost:8000 from your browser
2. Select your *.csv file that you want to be clustered.
3. Input a algorithm parameter. In this app you need to input following parameter:

    ```text
    K               = Don't use KNN or use KNN with K = 1 till 7 (recommended value)
    Alpha           = 0.1 till 1 (recommended value)
    Eta             = 0.1 till 1 (recommended value)
    Epoch           = minimum 30 is recommended
    Neuron Size     = 3x3, 4x4, 5x5 etc
    ```

4. After all parameter input is filled, click 'Mulai Clustering' to start clustering process.

## The result?

As you can see, the app show clustering progress and report alongside cluster visualization from epoch through epoch.

When clustering process is complete, you can see overall Silhouette Coefficient alongside with all member Silhouette Coefficient.

## Credit(s)

I don't built a webserver, built an array transformation algorithm or any code that doesn't relevant in my undergraduate thesis from scratch. You can check `environment.yml` to see what packages I used for this project.

## Usage

Yes, you can use some bunch of code in this repo to your personal project. Glad if you put a credit to this repo :)
