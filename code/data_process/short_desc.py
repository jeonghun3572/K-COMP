import os
import json
import xml.sax
import argparse
import subprocess
from src.util import WikiPageProcessor, WikiXmlHandler, TimeoutException, time_limit, generic_title_start


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    handler = WikiXmlHandler()
    wiki_page_processor = WikiPageProcessor()
    parser = xml.sax.make_parser()
    parser.setContentHandler(handler)

    num_processed_pages = 0
    last_shown_num_processed_pages = 0
    out_records = open(os.path.join(args.out_dir, 'articles.json'), "w")
    out_short_desc = open(os.path.join(args.out_dir, 'short_descriptions.json'), "w")

    for line in subprocess.Popen(['bzcat'], stdin=open(args.wiki_dump), stdout=subprocess.PIPE).stdout:
        parser.feed(line)
        if not handler.pages:
            continue

        # Clear finished pages to avoid keeping the whole dump in memory
        page = handler.pages[-1]
        is_redirect = page[1].startswith("#REDIRECT")
        num_processed_pages += len(handler.pages) - is_redirect
        handler.pages.clear()

        if not is_redirect:
            try:
                with time_limit(30):
                    record = wiki_page_processor.process(page)
                    if not list(filter(record['title'].startswith, generic_title_start)):
                        json.dump(record, out_records)
                        out_records.write('\n')
                        short_desc = record['short_description'] if record['short_description'] else record['first_sentence']
                        json.dump({record['title']: short_desc}, out_short_desc)
                        out_short_desc.write('\n')
            except TimeoutException:
                print("Timed out!", page[0])

        if 0 < args.max_articles < num_processed_pages:
            break
        if num_processed_pages % 1000 == 0 and num_processed_pages > last_shown_num_processed_pages:
            print("   Processed", num_processed_pages, "records")
            last_shown_num_processed_pages = num_processed_pages

    out_records.close()
    out_short_desc.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki-dump", type=str, required=True, help="Full path to the XML dump of Wikipedia with bz2 compression")
    parser.add_argument("--max-articles", type=int, default=-1, help="Maximum number of Wikipedia articles to process")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory where the output should be saved")
    args = parser.parse_args()
    main(args)
