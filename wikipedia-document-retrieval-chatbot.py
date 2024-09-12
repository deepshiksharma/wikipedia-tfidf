from sys import exit
from copy import deepcopy

from wikipediaapi import Wikipedia

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Check if required nltk resources are available; download resource if not found
nltk_resources = ['stopwords', 'punkt_tab', 'wordnet']
for resource in nltk_resources:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        print(f"{resource} not found. Downloading {resource}...")
        nltk.download(resource)
        print(f'{resource} downloaded.')

LEMMATIZER = WordNetLemmatizer()
STOPWORDS_ENG = stopwords.words('english')


def main():
    print('DOCUMENT RETRIEVAL BASED CHATBOT FOR WIKIPEDIA ARTICLES')
    print('(type "exit" to quit the program)\n')

    wiki_title, wiki_article = retrieve_article()
    original_sent_MASTER, processed_sent_MASTER = process_article(wiki_article)
    print('Text pre-processing completed.\n')
    
    print(f'Ask me questions about {wiki_title}')

    TfidfVec = TfidfVectorizer(stop_words=STOPWORDS_ENG)

    while True:
        user_query = input('Question:  ').rstrip('?')
        if user_query.casefold() == 'exit': exit('byee!')
        else:
            original_sent, processed_sent = deepcopy(original_sent_MASTER), deepcopy(processed_sent_MASTER)
            original_sent.append(user_query)

            processed_query = list()
            query_terms = word_tokenize(user_query)
            for term in query_terms:
                if term not in STOPWORDS_ENG:
                    processed_query.append(LEMMATIZER.lemmatize(term))
            processed_query = ' '.join(processed_query)
            processed_sent.append(processed_query)

            chatbot(original_=original_sent, processed_=processed_sent, vectorizer_=TfidfVec)


def retrieve_article():
    '''
    Prompts the user to input a Wikipedia article title and attempts to retrieve its text.
    Allows up to 3 attempts before defaulting to the "Python (programming language)" article.
    If the user types 'exit', the function terminates the program.

    Returns:
        tuple:
            wiki_title (str): title of the Wikipedia article that was fetched.
            wiki_article (str or None): text of the retrieved Wikipedia article.
    '''

    user_agent = 'RandomBot/0.0 (https://random.org/randombot/; randombot@random.org) generic-library/0.0'
    wiki = Wikipedia(language='en', user_agent=user_agent)

    i = 0
    while i < 3:
        wiki_title = input('Wikipedia article to retrieve text from:\n>  ').strip()
        if wiki_title.casefold() == 'exit': exit('byee!')

        retrieval_state, wiki_article = get_article_text(wiki, wiki_title)
        if retrieval_state:
            break
        else:
            i += 1

    if not retrieval_state:
        wiki_title = 'Python (programming language)'
        print(f'Attempts to fetch article failed 3 times, defaulting to fetch the Wikipedia article on "{wiki_title}"')
        retrieval_state, wiki_article = get_article_text(wiki, wiki_title)

    return wiki_title, wiki_article


def get_article_text(wiki_api_object, article_title):
    '''
    Fetches text from a Wikipedia article.

    Args:
        wiki_api_object: Wikipedia API object
        article_title (str): title of the Wikipedia article

    Returns:
        tuple:
            retrieval_state (bool): True if the article was fetched successfully, otherwise False.
            article_text (str or None): article text if retrieval was successful, otherwise None.
    '''
    
    retrieval_state, article_text = False, None

    try:
        page = wiki_api_object.page(article_title)
        if page.exists():
            article_text = page.text
            print(f'Successfully fetched the Wikipedia article on "{article_title}"')
            retrieval_state = True
        else:
            print(f'The article "{article_title}" does not exist, enter a valid Wikipedia article name.\n')
    
    except Exception as e:
        print(f'An error occurred while fetching the article:\n{e}')
    
    return retrieval_state, article_text


def process_article(article):
    # Sentence tokenization
    original_sentences = sent_tokenize(article)
    processed_sentences = deepcopy(original_sentences)

    # Stop-word removal and lemmatization
    for i in range(len(processed_sentences)):
        words = word_tokenize(processed_sentences[i])
        
        processed_words = list()
        for word in words:
            word = word.lower()
            if word not in STOPWORDS_ENG:
                processed_words.append(LEMMATIZER.lemmatize(word))
            elif word.isnumeric():
                processed_words.append(word)

        processed_sentence = ' '.join(processed_words) 
        processed_sentences[i] = processed_sentence

        return original_sentences, processed_sentences


def chatbot(original_, processed_, vectorizer_):
    tfidf = vectorizer_.fit_transform(processed_)

    similarities = cosine_similarity(tfidf[-1], tfidf)
    most_similar_idx = similarities.argsort()[0][-2]
    similarities_flat = similarities.flatten()
    similarities_flat.sort()
    most_similar_tfidf = similarities_flat[-2]

    print('Response:  ', end='')
    if most_similar_tfidf == 0:
        print('Unable to answer question.')
    else:
        print(original_[most_similar_idx])
    print(f'[most similar tf-idf: {most_similar_tfidf}]\n')


if __name__ == '__main__':
    main()
