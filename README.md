# Testing Playgroung

# Objective

The purpose of this repo is to experiment testing with an ML project. 
I'll be using **pytest**.

# Data and Model

For this project, the Titanic dataset is used.

The model is going to be a random forest. Note that no optimization is perform, nor advanced feature engineering.

# Reference

unittest + nose2
Unit Testing for Data Science with Python
https://towardsdatascience.com/unit-testing-for-data-science-with-python-16dfdcfe3232 

How to Develop and Test Your Google Cloud Function Locally
https://towardsdatascience.com/how-to-develop-and-test-your-google-cloud-function-locally-96a970da456f

How To Easily And Confidently Implement Unit Tests In Python
https://towardsdatascience.com/how-to-easily-and-confidently-implement-unit-tests-in-python-cad48d91ab74

THIS ONE
Testing features with pytest
https://towardsdatascience.com/testing-features-with-pytest-82765a13a0e7

How to do Unit Testing in dbt
https://towardsdatascience.com/how-to-do-unit-testing-in-dbt-cb5fb660fbd8

Unit Testing with Mocking in 10 Minutes
https://towardsdatascience.com/unit-testing-with-mocking-in-10-minutes-e28feb7e530

THIS ONE
An Elegant Guide to Testing Your Data Science Pipeline Using Pytest
https://towardsdatascience.com/an-elegant-guide-to-testing-your-data-science-pipeline-using-pytest-4859b0c32591

THIS ONE
Pytest for Data Scientists
https://towardsdatascience.com/pytest-for-data-scientists-2990319e55e6

Pytest with Marking, Mocking, and Fixtures in 10 Minutes
https://towardsdatascience.com/pytest-with-marking-mocking-and-fixtures-in-10-minutes-678d7ccd2f70

Pytest Tutorial
https://www.tutorialspoint.com/pytest/pytest_environment_setup.html

THIS ONE
13 Tips for using PyTest
https://towardsdatascience.com/13-tips-for-using-pytest-5341e3366d2d

PyTest for Machine Learning — a simple example-based tutorial
https://towardsdatascience.com/pytest-for-machine-learning-a-simple-example-based-tutorial-a3df3c58cf8

Testing Best Practices for Machine Learning Libraries
https://towardsdatascience.com/testing-best-practices-for-machine-learning-libraries-41b7d0362c95

How to get directory with test from fixture in conftest.py
https://medium.com/opsops/how-to-get-directory-with-test-from-fixture-in-conftest-py-275b566fcc00

Pytest: How to use fixtures as arguments in parametrize
https://engineeringfordatascience.com/posts/pytest_fixtures_with_parameterize/

https://www.tutorialspoint.com/pytest/index.htm

https://lyz-code.github.io/blue-book/coding/python/pytest/
https://docs.pytest.org/en/6.2.x/logging.html
https://docs.pytest.org/en/7.1.x/how-to/fixtures.html
https://docs.pytest.org/en/7.2.x/getting-started.html
https://docs.pytest.org/en/6.2.x/fixture.html


https://stackoverflow.com/questions/64672497/unit-testing-mock-gcs
https://cloud.google.com/functions/docs/samples/functions-storage-unit-test?hl=fr#functions_storage_unit_test-python
https://github.com/GoogleCloudPlatform/python-docs-samples/blob/HEAD/functions/helloworld/sample_storage_test.py
https://blog.engineering.publicissapient.fr/2020/09/18/comment-empecher-unittest-mock-de-se-moquer-de-vous/

https://pypi.org/project/pytest-bigquery-mock/

https://stackoverflow.com/questions/34466027/in-pytest-what-is-the-use-of-conftest-py-files


A useful tip when figuring out what's going wrong is to have some .py file to experiment. Of course, this file should 
not be Git-ed

It's useful for 2 reasons :
    - correcting the tested function or extend its ability to handle unexpected cases
    - correcting the test itself. After all, it is code :) . It might as well be a coding issue as a logical one 