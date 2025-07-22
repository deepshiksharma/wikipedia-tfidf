# Retrieval-Based Question Answering from Wikipedia Articles
Allows users to interactively ask questions about a specific Wikipedia article. 

Sentences from the article are converted into TF-IDF ([Term Frequency–Inverse Document Frequency][1]) vectors using [TfidfVectorizer][2].
[Cosine similarity][3] is then computed to find the sentence most similar to the user’s query.
The [Wikipedia API][4] is used to fetch text; [NLTK][5] is used for text preprocessing.

[1]: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
[2]: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
[3]: https://en.wikipedia.org/wiki/Cosine_similarity
[4]: https://pypi.org/project/Wikipedia-API/
[5]: https://www.nltk.org/

### nltk package dependencies
The required nltk packages for this program can be downloaded from the Python shell like this:
```py
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
```
