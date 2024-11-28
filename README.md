# Wikipedia Document Retrieval Chatbot
Document-retrieval-based chatbot for Wikipedia articles.

The [Wikipedia API][1] is used to retrieve text from Wikipedia articles.
[nltk][2] is used to pre-process text.
[TfidfVectorizer][3] is used to implement [Term Frequency–Inverse Document Frequency][4] on the processed text.
[Cosine similarity][5] is then used to compute the similarity between the user query and document vectors.

[1]: https://pypi.org/project/Wikipedia-API/
[2]: https://www.nltk.org/
[3]: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
[4]: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
[5]: https://en.wikipedia.org/wiki/Cosine_similarity

### nltk package dependencies
The required nltk packages for this program can be downloaded from the Python shell like this:
```py
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
```
