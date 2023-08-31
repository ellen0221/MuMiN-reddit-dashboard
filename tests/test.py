import re
import numpy as np
import gensim.downloader
from flask import (
    Blueprint, g, render_template, request, Response
)
from torchtext.data import get_tokenizer
from json import dumps
from dotenv import load_dotenv
import os
import io
import praw
from newspaper import Article, ArticleException, Config
from keras.models import load_model
from transformers import pipeline
from huggingface_hub import from_pretrained_keras
import urllib.request
from PIL import Image
load_dotenv()


def get_info_from_reddit(url):
    # Create a praw.Reddit instance
    reddit = praw.Reddit(
        client_id=os.getenv('REDDIT_CLIENT_ID'),
        client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
        refresh_token=os.getenv('REDDIT_REFRESH_TOKEN'),
        user_agent=os.getenv('REDDIT_USER_AGENT'),
    )
    # Search Submission by url
    submission = reddit.submission(url=url)
    content_text = submission.title + '\n'
    content_features = []
    context_text = None
    context_features = [submission.score]
    image = None
    if hasattr(submission, "selftext") and submission.selftext != '':
        content_text += submission.selftext
    elif hasattr(submission, "post_hint") and submission.post_hint == 'link':
        link = submission.url
        link = re.sub(r'\?.*', '', link)
        link = re.sub(r'\/$', '', link)
        article = crawlingArticle(link)
        if article is not None:
            content_text += article['doc']
            image = article['image']
    elif hasattr(submission, "post_hint") and submission.post_hint == 'image':
        image = submission.url
    else:
        return Response(dumps({'result': "Cannot analyse the content.", 'url': url}), mimetype='application/json')

    if image is not None:
        image_features = extract_image_feature(image)
        content_features = image_features
    else:
        content_features = np.zeros(6, dtype=float)

    content_features = np.array(content_features).reshape((-1, 1))
    if len(content_features) < 6:
        content_features = pad_embedding(np.array(content_features), 6)

    content_text = embed_by_GloVe(content_text, 500)

    tokenizer_kwargs = {'padding': True, 'truncation': True, 'max_length': 512}
    clf = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)
    # Emotion labels (See: https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
    emotion_dict = dict(anger=1, disgust=2, fear=3, joy=4, neutral=5, sadness=6, surprise=7)
    submission.comment_sort = "best"
    comment_list = submission.comments.list()[:5]
    for c in comment_list:
        text = embed_by_GloVe(c.body_html, 100)
        emotion = clf(c.body_html, **tokenizer_kwargs)[0][0]
        context_features.append(emotion_dict[emotion['label']])
        if text.size == 0:
            text = np.zeros((100, 100), dtype=float)
        text = pad_embedding(text, 100)
        context_text = text if context_text is None else np.concatenate((context_text, text))

    context_features = np.array(context_features).reshape((-1, 1))
    if len(context_features) < 6:
        context_features = pad_embedding(np.array(context_features), 6)

    # Padding and truncation
    content_text = pad_embedding(content_text, 500)
    context_text = pad_embedding(context_text, 500)

    res = get_prediction_from_trained_model(content_text, content_features, context_text, context_features)
    # print(res)



def get_prediction_from_trained_model(content_text, content_features, context_text, context_features):
    classes = ["Factual", "Misinformation"]
    # Load the local model from huggingface
    # model = from_pretrained_keras("ellen-0221/dl-reddit-misinformation-classifier")
    model = load_model('../misinformation_classifier.keras')
    model.summary()
    res = model.predict([[content_text], [content_features], [context_text], [context_features]])

    return classes[np.argmax(res)]


def pad_embedding(embedding: np.array, max_length: int):
    '''
    Pad or truncate the word embedding list to 'max_length'.

    Args:
        embedding: The word embedding list
        max_length: The max length which would be used to pad or truncate the word embedding list

    Returns:
        embedding: In the form of 'numpy.array'
    '''
    length = embedding.shape[0]
    if length > max_length:
        embedding = embedding[:max_length]
    elif length < max_length:
        embedding = np.concatenate((embedding, np.zeros(((max_length - length), embedding.shape[1]), dtype=float)))

    return np.array(embedding, dtype=float)


glove_wv = gensim.downloader.load('glove-wiki-gigaword-100')


def embed_by_GloVe(doc: str, max_length: int = -1):
    '''
    Tokenise and embed the sentence with GloVe

    Args:
        doc: The sentence needed to be embedded.
        max_length: The max length of the return word embedding list
    Returns:
        A list of word embedding.
    '''
    tokenizers = get_tokenizer('basic_english')
    tokenized_doc = tokenizers(doc)
    embedding = []
    for word in tokenized_doc:
        if word in glove_wv:
            embedding.append(glove_wv[word])
        if max_length != -1:
            if len(embedding) == max_length:
                break

    return np.array(embedding)


def extract_image_feature(imageUrl):
    try:
        if '.gif' not in imageUrl:
            # Get image's size, width, and height
            file = urllib.request.urlopen(imageUrl)
            im = Image.open(file)
            # image_byte = image_to_byte_array(im)
            # image_properties = extract_dominant_color(image_byte)
            # image_properties = {'width': im.width, 'height': im.height,
            #                     'size': im.width * im.height}
            image_properties = [im.width, im.height, im.width * im.height, 0, 0, 0]
            return image_properties
    except:
        return [0, 0, 0, 0, 0, 0]


def image_to_byte_array(image: Image) -> bytes:
    # BytesIO is a file-like buffer stored in memory
    imgByteArr = io.BytesIO()
    # image.save expects a file-like as a argument
    image.save(imgByteArr, format=image.format)
    # Turn the BytesIO object back into a bytes object
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


def crawlingArticle(url):
    # Initialise article
    try:
        article = Article(url)
        article = article.download()
        if article is None:
            return None
        article.parse()
    except (ArticleException, ValueError, RuntimeError, TimeoutError):
        return None

    # Extract the title and skip URL if it is empty
    title = article.title
    title = re.sub('\n+', '\n', title)
    title = re.sub(' +', ' ', title)
    title = title.strip()

    # Extract the content and skip URL if it is empty
    content = article.text.strip()
    content = re.sub('\n+', '\n', content)
    content = re.sub(' +', ' ', content)
    content = content.strip()

    doc = title + '\n' + content

    # Extract top image
    top_image = article.top_image

    return {'title': title, 'content': content,
            'doc': doc, 'image': top_image}


get_info_from_reddit("https://www.reddit.com/r/stupidpol/comments/v1gawp/oprf_to_implement_racebased_grading_system_in/")
# crawlingArticle('https://westcooknews.com/stories/626581140-oprf-to-implement-race-based-grading-system-in-2022-23-school-year')
