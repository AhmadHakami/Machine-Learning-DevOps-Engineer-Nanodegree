# Production Ready Code

## Catching Errors

To ensure smooth code execution regardless of the function arguments, it is crucial to identify and handle any potential errors. This can be achieved using the `try` and `except` methods.

## Testing

Effective code should meet the following conditions, serving as a basis for testing:

- It should be automated
- The execution should be fast
- The code should be reliable
- The errors and logs should be informative
- The testing should be focused

## Unit Testing Tools

- PyTest

The `pytest` framework simplifies writing small, readable tests. It can also scale to support complex functional testing for applications and libraries.

## Logging Messages

Logging is the process of recording messages to describe the events occurring during the execution of your software.

### Logging Tips

- Be professional and clear  
  Bad examples: `Hmmm... this isn't working???`, `idk.... :(`  
  Good example: `Couldn't parse file.`  

- Be concise and use normal capitalization  
  Bad examples: `Start Product Recommendation Process`, `We have completed the steps necessary and will now proceed with the recommendation process for the records in our product database.`  
  Good example: `Generating product recommendations.`  

- Choose the appropriate level for logging  
    - **_Debug_**: Use this level for anything that occurs in the program.
    - **_Error_**: Use this level to record errors.
    - **_Info_**: Use this level to record all user-driven or system-specific actions, such as regularly scheduled operations.

- Provide any useful information  
  Bad example:  `Failed to read location data`  
  Good example: `Failed to read location data: store_id 8324971`  

## Additional Reading Material

1. Python tutorial for `try` and `except` methods: [Errors and Exceptions](https://docs.python.org/3/tutorial/errors.html).
2. Four Ways Data Science Goes Wrong and How Test-Driven Data Analysis Can Help: [Blog Post](https://www.predictiveanalyticsworld.com/patimes/four-ways-data-science-goes-wrong-and-how-test-driven-data-analysis-can-help/6947/).
3. Getting Started with Testing by Ned Batchelder: [Slide Deck](https://speakerdeck.com/pycon2014/getting-started-testing-by-ned-batchelder), [Presentation Video](https://www.youtube.com/watch?v=FxSsnHeWQBY).
4. Understanding how to pass parameters to the fixture functions using the built-in request object: [Parametrizing fixtures](https://docs.pytest.org/en/7.1.x/how-to/fixtures.html#fixture-parametrize).
5. Additional resources on testing using Google's cloud-based systems: [Testing Debugging](https://developers.google.com/machine-learning/testing-debugging/pipeline/deploying).
6. Why Test-Driven Development (TDD) is Essential for Good Data Science: [Here's Why](https://medium.com/@karijdempsey/test-driven-development-isessential-for-good-data-science-heres-why-db7975a03a44).
7. For testing ML models in production settings, it is commonly done using cloud-based software. [here on testing using Google's cloud-based systems](https://developers.google.com/machine-learning/testing-debugging/pipeline/deploying).