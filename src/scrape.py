import common
from log import log, warning, error, important, set_status_state, ProgressStatus

import subprocess
import os
import json
import re
import time
import random
from bs4 import BeautifulSoup

dataset_post_dir = os.path.join(common.dataset_dir, "posts")
common.ensure_dir(dataset_post_dir)


def wait_for_file(filename):
    max_checks = 20
    while not os.path.isfile(filename):
        max_checks -= 1
        if max_checks <= 0:
            raise RuntimeError("loading file was too slow")
        time.sleep(1)


def courtesy_delay():
    time.sleep(random.randint(10, 200) / 20)


def is_new_topic(topic_id):
    return topic_id.replace("/", "_") + ".json" not in [
        os.path.basename(filename)
        for filename in common.list_all_files(dataset_post_dir)
    ]


def load_new_topics():
    log("Loading recommended page")
    url = f"https://vauva.fi/?sort=recommended"
    html_path = os.path.join(common.cache_dir, common.timestamp())
    subprocess.check_output(["wget", url, "-O", html_path], stderr=subprocess.DEVNULL)

    wait_for_file(html_path)
    page = BeautifulSoup(
        common.read_file_to_string(html_path).strip(), features="html.parser"
    )

    def parse_link(link: str):
        link = link["href"]
        match = re.match(r"^/keskustelu/([0-9]*)/([^\? ]*)", link)
        return match and f"{match.group(1)}/{match.group(2)}"

    links = page.find_all("a", href=True)
    topics = [parse_link(link) for link in links]
    topics = list(set(topic for topic in topics if topic and is_new_topic(topic)))

    return topics


def scrape_page(topic_id, page):
    courtesy_delay()

    log(f"Loading page {page:2.0f}", repeating_status=True)
    url = f"https://vauva.fi/keskustelu/{topic_id}?page={page}"
    html_path = os.path.join(common.cache_dir, common.timestamp())
    try:
        subprocess.check_output(
            ["wget", url, "-O", html_path], stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        error("Failed to load page")
        return []

    wait_for_file(html_path)
    page = BeautifulSoup(
        common.read_file_to_string(html_path).strip(), features="html.parser"
    )
    comments = page.find_all("article", {"class": "comment"})
    text = [
        el.text.strip()
        for comment in comments
        for el in comment.find("div", {"class": "middle"})
        .find("div", {"class": "field-item"})
        .find_all("p")
    ]
    return text


def scrape_topic(topic_id):
    important(f"Starting to load topic: {topic}")
    url = f"https://vauva.fi/keskustelu/{topic_id}"
    html_path = os.path.join(common.cache_dir, common.timestamp())
    try:
        subprocess.check_output(
            ["wget", url, "-O", html_path], stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        warning("Failed to load first page")
        return

    wait_for_file(html_path)
    page = BeautifulSoup(
        common.read_file_to_string(html_path).strip(), features="html.parser"
    )
    link_to_last = page.find("li", {"class": "pager-last"})
    link_to_last = link_to_last.find("a", href=True)
    num_pages = int(re.match(".*page=([0-9]*)", link_to_last["href"]).group(1)) + 1
    log(f"Pages in topic: {num_pages}")

    set_status_state(ProgressStatus(num_pages))
    posts = [post for i in range(num_pages) for post in scrape_page(topic_id, i)]

    path = os.path.join(dataset_post_dir, topic_id.replace("/", "_") + ".json")
    with open(path, "w") as file:
        json.dump(posts, file)
    log(f"Saved {len(posts)} posts")


if __name__ == "__main__":
    important("Scraping quality posts")
    topics = load_new_topics()
    important(f"Found {len(topics)} new topics")
    for topic in topics:
        scrape_topic(topic)
