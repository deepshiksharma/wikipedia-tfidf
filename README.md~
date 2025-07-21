# Retrieval-Based Question Answering from Wikipedia Articles
Allows users to interactively ask questions about a specific Wikipedia article. 

The [Wikipedia API][1] is used to retrieve text, preprocessed using [nltk][2].
Using [TfidfVectorizer][3], sentences from the article are converted into TF-IDF ([Term Frequency–Inverse Document Frequency][4]) vectors. [Cosine similarity][5] is then computed to find the sentence most similar to the user’s query.

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
