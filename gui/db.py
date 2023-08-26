import click
from neo4j import GraphDatabase
from flask import current_app, g
from dotenv import load_dotenv
import os
load_dotenv()

URI = "bolt://localhost:7687"
AUTH = (os.getenv('NEO4J_REDDIT_USER'), os.getenv('NEO4J_REDDIT_PASSWORD'))


def get_db():
    if 'db' not in g:
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            driver.verify_connectivity()
            g.db = driver
    return g.db


def close_db(e=None):
    db = g.pop('db', None)

    if db is not None:
        db.close()


def init_db():
    db = get_db()


@click.command('init-db')
def init_db_command():
    init_db()
    click.echo('Test the database')


def init_app(app):
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)
