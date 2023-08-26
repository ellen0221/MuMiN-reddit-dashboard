import re
import time
import numpy as np
import gensim
from flask import (
    Blueprint, g, render_template, request, Response
)
from textwrap import dedent
from typing import cast
from typing_extensions import LiteralString
from torchtext.data import get_tokenizer
import datetime
from json import dumps
from gensim.parsing.preprocessing import STOPWORDS
from dotenv import load_dotenv
import os
import io
import praw
from newspaper import Article, ArticleException
from transformers import BartTokenizer, BartForConditionalGeneration
import urllib.request
from PIL import Image
from gui.db import get_db

load_dotenv()

bp = Blueprint('dashboard', __name__, url_prefix='/dashboard')


def query(q: LiteralString) -> LiteralString:
    # this is a safe transform:
    # no way for cypher injection by trimming whitespace
    # hence, we can safely cast to LiteralString
    return cast(LiteralString, dedent(q).strip())


query_all_mis_reddit_date = '''
    MATCH (n:Reddit)-[:SIMILAR|SIMILAR_VIA_ARTICLE]-(c:Claim)
    WITH n, c
    MATCH (c)-[:HAS_LABEL]->(l:Label)
    WHERE l.verdict='misinformation'
    RETURN n.createdAt AS time
'''

query_all_fac_reddit_date = '''
    MATCH (n:Reddit)-[:SIMILAR|SIMILAR_VIA_ARTICLE]-(c:Claim)
    WITH n, c
    MATCH (c)-[:HAS_LABEL]->(l:Label)
    WHERE l.verdict='factual'
    RETURN n.createdAt AS time
'''

query_mis_reddit_date_conditioned = '''
    MATCH (n:Reddit)-[:SIMILAR|SIMILAR_VIA_ARTICLE]-(c:Claim)
    WHERE n.createdAt >= $start_time and n.createdAt <= $end_time
    WITH n, c
    OPTIONAL MATCH (c)-[:HAS_LABEL]->(l:Label)
    WHERE l.verdict='misinformation'
    RETURN n.createdAt AS time
'''

query_fac_reddit_date_conditioned = '''
    MATCH (n:Reddit)-[:SIMILAR|SIMILAR_VIA_ARTICLE]-(c:Claim)
    WHERE n.createdAt >= $start_time and n.createdAt <= $end_time
    WITH n, c
    OPTIONAL MATCH (c)-[:HAS_LABEL]->(l:Label)
    WHERE l.verdict='factual'
    RETURN n.createdAt AS time
'''

query_mis_reddit_graph = '''
    MATCH (n:Reddit)-[:SIMILAR|SIMILAR_VIA_ARTICLE]-(c:Claim)
    WITH n, c
    MATCH (c)-[:HAS_LABEL]->(l:Label)
    WHERE l.verdict='misinformation'
    WITH n
    MATCH (n)-[:BELONGS_TO]->(s:Subreddit)
    MATCH (u:User)-[:POSTED]->(n)
    RETURN n.redditId AS reddit, s.name as subreddit, u.userFullname as user LIMIT $limit
'''

query_fac_reddit_graph = '''
    MATCH (n:Reddit)-[:SIMILAR|SIMILAR_VIA_ARTICLE]-(c:Claim)
    WITH n, c
    MATCH (c)-[:HAS_LABEL]->(l:Label)
    WHERE l.verdict='factual'
    WITH n
    MATCH (n)-[:BELONGS_TO]->(s:Subreddit)
    MATCH (u:User)-[:POSTED]->(n)
    RETURN n.redditId AS reddit, s.name as subreddit, u.userFullname as user LIMIT $limit
'''

query_mis_reddit_content = '''
    MATCH (n:Reddit)-[:SIMILAR|SIMILAR_VIA_ARTICLE]-(c:Claim)
    WITH n, c
    MATCH (c)-[:HAS_LABEL]->(l:Label)
    WHERE l.verdict='misinformation'
    WITH n
    OPTIONAL MATCH (n)-[:HAS_ARTICLE]->(a:Article)
    RETURN n.title AS title, n.text as text, a.summary as article LIMIT $mis_limit
'''

query_fac_reddit_content = '''
    MATCH (n:Reddit)-[:SIMILAR|SIMILAR_VIA_ARTICLE]-(c:Claim)
    WITH n, c
    MATCH (c)-[:HAS_LABEL]->(l:Label)
    WHERE l.verdict='factual'
    WITH n
    OPTIONAL MATCH (n)-[:HAS_ARTICLE]->(a:Article)
    RETURN n.title AS title, n.text as text, a.summary as article LIMIT $fac_limit
'''


@bp.route('/')
def index():
    return render_template('dashboard.html')


@bp.route('/classifier', methods=["POST"])
def classifier():
    url = request.form.get("url", type=str)

    # Create a praw.Reddit instance
    # reddit = praw.Reddit(
    #     client_id=os.getenv('REDDIT_CLIENT_ID'),
    #     client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    #     refresh_token=os.getenv('REDDIT_REFRESH_TOKEN'),
    #     user_agent=os.getenv('REDDIT_USER_AGENT'),
    # )
    # # Search Submission by url
    # submission = reddit.submission(url=url)
    # title = submission.title
    # content = None
    # features = [submission['ups']]
    # image = None
    # if hasattr(submission, "selftext"):
    #     content = submission['selftext']
    # elif hasattr(submission, "post_hint") and submission.post_hint == 'link':
    #     link = submission['url']
    #     link = re.sub(r'\?.*', '', link)
    #     link = re.sub(r'\/$', '', link)
    #     article = crawlingArticle(link)
    #     content = article['summary']
    #     image = article['image']
    # elif hasattr(submission, "post_hint") and submission.post_hint == 'image':
    #     image = submission['url']
    #     image_features = extract_image_feature(image)
    #     features = features.extend(image_features)
    #
    # text = title + '\n' + content
    #
    # submission.comment_limit = 5
    # for c in list(submission.comments):
    #     text += '\n' + c['body_html']
    #
    # res = get_prediction_from_trained_model(text, features)

    return Response(dumps({'result': 1, 'url': url}), mimetype='application/json')


def get_prediction_from_trained_model(text: str, features: list):
    embed_text = embed_by_GloVe(text)
    embed_text = pad_embedding(embed_text, 1000)
    if len(features) != 7:
        features += 0*(7-len(features))
    features = np.array(features).reshape((-1, 1))
    # model =

    return 1


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


def embed_by_GloVe(doc: str, max_length: int = -1):
    '''
    Tokenise and embed the sentence with GloVe

    Args:
        doc: The sentence needed to be embedded.
        max_length: The max length of the return word embedding list
    Returns:
        A list of word embedding.
    '''
    glove_wv = gensim.downloader.load('glove-wiki-gigaword-100')
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
            image_byte = image_to_byte_array(im)
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

    # Load the summarisation model and its tokeniser
    transformer = 'facebook/bart-large-cnn'
    tokeniser = BartTokenizer.from_pretrained(transformer)
    model = BartForConditionalGeneration.from_pretrained(transformer)
    tokens = tokeniser([doc], return_tensors='pt', padding=True,
                       truncation=True, max_length=1000)

    summary_ids = model.generate(tokens['input_ids'],
                                 num_beams=4,
                                 max_length=512,
                                 early_stopping=True)

    summary = tokeniser.batch_decode(
        summary_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    # Extract top image
    top_image = article.top_image

    return {'title': title, 'content': content,
            'summary': summary, 'image': top_image}


@bp.route('/countRedditByDate')
def countRedditByDate():
    try:
        startDate = request.args.get("start_date", type=str)
        endDate = request.args.get("end_date", type=str)
    except KeyError:
        return []
    else:
        db = get_db()
        if startDate != '' and endDate != '':
            startTime = time.mktime(datetime.datetime.strptime(startDate, "%Y-%m").timetuple())
            endTime = time.mktime(datetime.datetime.strptime(endDate, "%Y-%m").timetuple())
            mis_records, _, _ = db.execute_query(
                query(query_mis_reddit_date_conditioned),
                start_time=startTime,
                end_time=endTime
            )
            fac_records, _, _ = db.execute_query(
                query(query_fac_reddit_date_conditioned),
                start_time=startTime,
                end_time=endTime
            )
        else:
            mis_records, _, _ = db.execute_query(
                query(query_all_mis_reddit_date),
            )
            fac_records, _, _ = db.execute_query(
                query(query_all_fac_reddit_date),
            )

        # Convert timestamp to the 'yyyy-MM' format.
        year_month_format = '%Y-%m'
        mis_date_list = [datetime.datetime.fromtimestamp(r['time']).strftime(year_month_format) for r in mis_records]
        fac_date_list = [datetime.datetime.fromtimestamp(r['time']).strftime(year_month_format) for r in fac_records]
        # Count the posts with the same year and month.
        mis_date_count = count_str_list_to_dict(mis_date_list, 'x', 'y', 0)
        fac_date_count = count_str_list_to_dict(fac_date_list, 'x', 'y', 0)
    return Response(
        dumps({'mis': mis_date_count, 'fac': fac_date_count,
               'misCount': len(mis_records), 'facCount': len(fac_records)}),
        mimetype="application/json"
    )


def count_str_list_to_dict(date_list: list, str_key: str, count_key: str,
                           sort: int = 1, reverse: bool = False):
    counts = {}
    for i in date_list:
        if i in counts.keys():
            counts[i] += 1
        else:
            counts[i] = 1
    # Sort by key
    sorted_counts = sorted(counts.items(), key=lambda x: x[sort], reverse=reverse)
    # Convert the dict to a form that the line chart can use.
    res = [{str_key: k, count_key: v} for (k, v) in sorted_counts]

    return res


@bp.route('/graph')
def getRedditGraph():
    limit = request.args.get("limit", type=int)
    db = get_db()
    mis_records, _, _ = db.execute_query(
        query(query_mis_reddit_graph),
        limit=limit
    )
    fac_records, _, _ = db.execute_query(
        query(query_fac_reddit_graph),
        limit=limit
    )
    mis_nodes = []
    mis_rel = []
    fac_nodes = []
    fac_rel = []
    for record in mis_records:
        mis_nodes.extend([{"id": record['reddit'], "label": "reddit"},
                          {"id": record['user'], "label": "user"},
                          {"id": record['subreddit'], "label": "subreddit"}])
        mis_rel.extend([{"source": record['user'], "target": record['reddit']},
                        {"source": record['reddit'], "target": record['subreddit']}])
    for record in fac_records:
        fac_nodes.extend([{"id": record['reddit'], "label": "reddit"},
                          {"id": record['user'], "label": "user"},
                          {"id": record['subreddit'], "label": "subreddit"}])
        fac_rel.extend([{"source": record['user'], "target": record['reddit']},
                        {"source": record['reddit'], "target": record['subreddit']}])

    return Response(dumps({"nodes": {"mis": mis_nodes, "fac": fac_nodes},
                           "links": {"mis": mis_rel, "fac": fac_rel}}),
                    mimetype="application/json")


@bp.route('/word-cloud')
def wordCloud():
    mis_limit = request.args.get("mis_limit", type=int)
    fac_limit = request.args.get("fac_limit", type=int)
    db = get_db()
    mis_records, _, _ = db.execute_query(
        query(query_mis_reddit_content),
        mis_limit=mis_limit
    )
    fac_records, _, _ = db.execute_query(
        query(query_fac_reddit_content),
        fac_limit=fac_limit
    )

    # Concatenating sentences
    mis_doc = ''
    for record in mis_records:
        mis_doc += record['title']
        mis_doc += record['text'] if record['text'] is not None else ''
        mis_doc += record['article'] if record['article'] is not None else ''
    mis_doc = clean_text(mis_doc)
    fac_doc = ''
    for record in fac_records:
        fac_doc += record['title']
        fac_doc += record['text'] if record['text'] is not None else ''
        fac_doc += record['article'] if record['article'] is not None else ''
    fac_doc = clean_text(fac_doc)

    # Tokenize the doc
    tokenizer = get_tokenizer('basic_english')
    mis_tokens = tokenizer(mis_doc)
    fac_tokens = tokenizer(fac_doc)
    # Remove the stop words and words with single letter
    mis_tokens_filtered = [w for w in mis_tokens if not w.lower() in STOPWORDS and len(w) > 1]
    fac_tokens_filtered = [w for w in fac_tokens if not w.lower() in STOPWORDS and len(w) > 1]
    mis_words = count_str_list_to_dict(mis_tokens_filtered, 'text', 'size', 1, True)[:200]
    fac_words = count_str_list_to_dict(fac_tokens_filtered, 'text', 'size', 1, True)[:200]

    # Calculate the font size based on the word count
    font_size_range = 65
    max_font_size = 80
    mis_min_count = mis_words[-1]['size']
    mis_max_count = mis_words[0]['size']
    fac_min_count = fac_words[-1]['size']
    fac_max_count = fac_words[0]['size']
    mis_font_ratio = font_size_range / (mis_max_count - mis_min_count)
    fac_font_ratio = font_size_range / (fac_max_count - fac_min_count)
    for idx, item in enumerate(mis_words[1:]):
        mis_words[idx]['size'] = max_font_size - ((mis_max_count - item['size']) * mis_font_ratio)
    for idx, item in enumerate(fac_words[1:]):
        fac_words[idx]['size'] = max_font_size - ((fac_max_count - item['size']) * fac_font_ratio)
    mis_words[0]['size'] = max_font_size
    fac_words[0]['size'] = max_font_size

    return Response(dumps({"mis": mis_words, "fac": fac_words,
                           "mis_count": len(mis_records), "fac_count": len(fac_records)}),
                    mimetype="application/json")


def clean_text(text: str):
    mention_regex = r'@[a-zA-Z0-9]*'
    url_regex = r'[\[]*[a-zA-Z@ ]*[\](]*http[a-zA-Z0-9.\/?&:%=\-\[\]\)]*'
    space_regex = r' {2,}'
    text = re.sub(mention_regex, ' ', text)
    text = re.sub(url_regex, ' ', text)
    text = re.sub(space_regex, ' ', text)
    return text
