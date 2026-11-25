# Project Notes

## Design choices and trade-offs

I used FastAPI, SQLAlchemy and SQLite to keep the application relatively simple while still demonstrating a proper API and database-backed workflow. The application supports both customer ticket submission and a team interface for reviewing and updating tickets.

For classification, I used TF-IDF features with Logistic Regression. This is lightweight, fast and suitable for the Bitext dataset. I used grouped splitting to reduce the risk of similar messages appearing in both training and test sets.

For FAQ retrieval, I used TF-IDF similarity instead of introducing a vector database and embedding infrastructure. With only 15 FAQ entries, this was sufficient for the demonstration. The LLM generates replies using the retrieved FAQ content, and the application displays the supporting sources.

I also included Docker Compose to make the application easier to run. I chose simpler implementations over more complex production infrastructure because the main goal was to demonstrate the required capabilities within the available time.

## AI coding tool mistakes

1. During manual testing, I found that refreshing the customer submission success page could resubmit the original POST request and create a duplicate ticket. Automated tests had not caught this browser behaviour.

2. Claude Code generated code that could raise an unhandled TypeError when used_faq_ids contained objects rather than strings. This was indentified by an independent read-only test reviewer.

These examples reminded me that AI-generated commands and files need to be checked rather than accepted without verification.

## What I would improve with another week

- Test the classifier with a larger collection of independently written, real-world customer messages.
- Improve FAQ retrieval and evaluate its relevance threshold using more test cases.
- Add background processing for reply generation so ticket submission does not depend on the LLM response time.
- Improve authentication, database migrations and application monitoring.
- Expand evaluation to measure classification quality and generated reply accuracy on a larger dataset.