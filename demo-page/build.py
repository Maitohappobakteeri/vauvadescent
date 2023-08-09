import os
import sys
import chevron
import html
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from log import log, LogTypes
import common
from predict import predict, initialize_for_predict

log("Building demo page")

model, device, config = initialize_for_predict()
input_text_list  = [
    "Miksi nykyään ihmiset ei enää",
    "Auttakaa! Mitä teen, kun mun naapurit ei lopeta ees yöllä ja en saa nukuttua",
    "Syönkö liikaa suklaata, kun",
    "Uskotko tähän uuteen juttuun vai et"
]

predictions = [
    predict(model, device, config, i, split_to_posts=True, max_predict_chars=1000)
    for i in input_text_list
]

def to_post_elemenet(post):
    title = "Vierailija:"
    body = html.escape(post)
    return f"<div class=\"card columns\"> <img class=\"card-image\"/> <div class=\"rows\"> <p class=\"card-title\"> {title} </p>  <p class=\"card-body\"> {body} </p> </div>  </div>"

def render_topic(i, topic):
    rendered = chevron.render(common.read_file_to_string("template.html"), { 
        'topic0': f"{input_text_list[0][:20]}...",
        'topic1': f"{input_text_list[1][:20]}...",
        'topic2': f"{input_text_list[2][:20]}...",
        'topic3': f"{input_text_list[3][:20]}...",
        'test': " ".join([to_post_elemenet(p) for p in topic])})
    with open(f"topic-{i}.html", "w") as file:
        file.write(rendered)

for (i, topic) in enumerate(predictions):
    render_topic(i, topic)